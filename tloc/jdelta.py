import numpy as np
from ase.io import read
from shutil import copyfile
from tloc.javerage import get_orbitals, catnip
from tloc import chdir, mkdir
import json
import os

def load_phonons(file='mesh.yaml'):
    import yaml
    from yaml import CLoader as Loader

    freqs = []
    vecs = []
    with open(file) as f:
        data = yaml.load(f, Loader=Loader)
    for phonon in data['phonon']:
        for mode in phonon['band']:
            freqs.append(mode['frequency'])
            vecs.append(mode['eigenvector'])
    
    # e modes, a atoms, v directions
    freqs_e = np.array(freqs)
    vecs_eav = np.array(vecs)

    return freqs_e, vecs_eav

def derivative(jpp, jp, jm, jmm, delta):
    return (-jpp + 8 * jp - 8 * jm + jmm) / 6 * delta

def get_dj_matrix(jlists, delta):
    latoms = len(jlists[0]) / 6

    # array with j + full delta (j plus plus)
    jpp = np.empty([latoms, 3])
    jpp[:, 0] = jlists[1][1::6]
    jpp[:, 1] = jlists[1][3::6]
    jpp[:, 2] = jlists[1][5::6]

    # array with j + half delta (j plus)
    jp = np.empty([latoms, 3])
    jp[:, 0] = jlists[0][1::6]
    jp[:, 1] = jlists[0][3::6]
    jp[:, 2] = jlists[0][5::6]
    
    # array with j - half delta (j minus)
    jm = np.empty([latoms, 3])
    jm[:, 0] = jlists[0][0::6]
    jm[:, 1] = jlists[0][2::6]
    jm[:, 2] = jlists[0][4::6]

    # array with j - full delta (j minus minus)
    jmm = np.empty([latoms, 3])
    jmm[:, 0] = jlists[1][0::6]
    jmm[:, 1] = jlists[1][2::6]
    jmm[:, 2] = jlists[1][4::6]

    dj_matrix = derivative(jpp, jp, jm, jmm, delta)

    return dj_matrix

def get_deviation(dj_av, phonon_file, temp):
    na = len(dj_av)
    freqs_e, vecs_eav = load_phonons(phonon_file)
    epcoup_e = np.einsum('av,eav->e', dj_av, vecs_eav)
    ssigma = (1 / na) * np.sum(epcoup_e**2 / \
        (2 * np.tanh(freqs_e / (2 * temp))))

    return np.sqrt(ssigma)

def get_displacements(atoms, all=True):
    if all:
        latoms = len(atoms)
    else:
        latoms = len(atoms) // 2
    for ia in range(latoms):
        for iv in range(3):
            for sign in [-1, 1]:
                yield (ia, iv, sign)

def displace_atom(atoms, ia, iv, sign, delta):
    new_atoms = atoms.copy()
    pos_av = new_atoms.get_positions()
    pos_av[ia, iv] += sign * delta
    new_atoms.set_positions(pos_av)
    return new_atoms

def finite_dif(delta, atoms, disp):
    ia = disp[0]
    iv = disp[1]
    sign = disp[2]
    prefix = 'dj-{}-{}{}{}' .format(int(delta * 1000), 
                                    ia,
                                    'xyz'[iv],
                                    ' +-'[sign])
    new_structure = displace_atom(atoms, ia, iv, sign, delta)
    mkdir(prefix)
    with chdir(prefix):
        get_orbitals(new_structure, prefix)
        
def multi_finite_dif(delta=0.01, all=True):
    from multiprocessing import Pool
    from functools import partial

    atoms = read('static.xyz')
    command = partial(finite_dif, delta, atoms)

    disps = []
    for ia, iv, sign in get_displacements(atoms, all=all):
        disps.append((ia, iv, sign))

    with Pool(processes=64) as pool:
        pool.map(command, disps)

def run_jdelta(pair, delta=0.01):
    # run Gaussian for displacements of first molecule
    mol1 = str(pair[1][0] + 1)
    with chdir(mol1):
        if not os.path.isdir('displacements'):
            mkdir('displacements')
            with chdir('displacements'):
                copyfile('../' + mol1 + '.xyz', 'static.xyz')
                finite_dif(delta / 2)
                finite_dif(delta)

    # run Gaussian for displacements of first molecule in the pair
    molpair = str(pair[0])
    with chdir(molpair):
        mkdir('displacements')
        with chdir('displacements'):
            copyfile('../' + molpair + '.xyz', 'static.xyz')
            finite_dif(delta / 2, all=False)
            finite_dif(delta, all=False)

def read_jdelta(delta=0.01, phonon_file='mesh.yaml', temp=0.025):
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)

    for pair in pairs.items():
        mol1 = str(int(pair[1][0]) + 1)
        mol2 = str(int(pair[1][1]) + 1)
        molpair = pair[0]

        # calculating j for each displacement
        path1 = mol1 + '/' + mol1 + '/displacements/'
        path2 = mol2 + '/' + mol2
        path3 = molpair + '/' + molpair + '/displacements/'

        jlists = []
        atoms = read(path1 + 'static.xyz')
        for d in [delta/2, delta]:
            js = []
            for ia, iv, sign in get_displacements(atoms, all=all):
                prefix = 'dj-{}-{}{}{}' .format(int(d * 1000), 
                                                    ia,
                                                    'xyz'[iv],
                                                    ' +-'[sign])
            
                j = catnip(path1 + prefix + '/' + prefix, path2, path3 + prefix + '/' + prefix)
                js.append(j)
            jlists.append(js)

        # Create GradJ matrix with a atoms and v directions
        dj_matrix_av = get_dj_matrix(jlists, delta)

        # Calculate jdelta
        jdelta = get_deviation(dj_matrix_av, phonon_file, temp)
        data = {molpair: jdelta}
        with open('DeltaJ_' + molpair + '.json', 'w', encoding='utf-8') as f:
             json.dump(data, f, ensure_ascii=False, indent=4)

def jdelta():
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)
    
    for pair in pairs.items():
        run_jdelta(pair, delta=0.01)

if __name__ == '__main__':
    jdelta()