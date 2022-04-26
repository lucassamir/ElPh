import numpy as np
from ase.io import read
from shutil import copyfile
from tloc.javerage import get_orbitals, catnip
from tloc import chdir, mkdir
import json
import os

def load_phonons(phonon_file='phonon.npz', map_file='atom_mapping.json'):
    # read phonon modes file
    phonon = np.load(phonon_file)

    # read mapping file
    with open(map_file, 'r') as json_file:
        map = json.load(json_file)
    
    # e modes, a atoms, v directions
    freqs_e = phonon['freqs'].flatten()
    vecs_eav = phonon['vecs'].real.reshape(len(freqs_e), -1, 3)

    # use mapping to order the wrapped phonon modes
    # based on the unwrapped atoms
    vecs_eav = vecs_eav[:, map, :]

    return freqs_e, vecs_eav

def get_dj_matrix(jlists, delta):
    latoms = len(jlists[0]) / 6

    # array with j - delta (j minus)
    jm = np.empty([latoms, 3])
    jm[:, 0] = jlists[0::6]
    jm[:, 1] = jlists[2::6]
    jm[:, 2] = jlists[4::6]

    # array with j + delta (j plus)
    jp = np.empty([latoms, 3])
    jp[:, 0] = jlists[1::6]
    jp[:, 1] = jlists[3::6]
    jp[:, 2] = jlists[5::6]

    dj_matrix = (jp - jm) / (2 * delta)

    return dj_matrix

def get_deviation(dj_av, phonon_file, temp):
    na = len(dj_av)
    freqs_e, vecs_eav = load_phonons(phonon_file)
    epcoup_e = np.einsum('av,eav->e', dj_av, vecs_eav)
    ssigma = (1 / na) * np.sum(epcoup_e**2 / \
        (2 * np.tanh(freqs_e / (2 * temp))))

    return np.sqrt(ssigma)

def get_displacements(atoms):
    latoms = len(atoms)
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

def read_finite_dif(delta, path1, path2, path3, offset, disp):
    ia = disp[0]
    iv = disp[1]
    sign = disp[2]
    if offset:
        prefix_mol = 'dj-{}-{}{}{}' .format(int(delta * 1000), 
                                            ia,
                                            'xyz'[iv],
                                            ' +-'[sign])
        prefix_pair = 'dj-{}-{}{}{}' .format(int(delta * 1000), 
                                             ia + offset,
                                             'xyz'[iv],
                                             ' +-'[sign])
        j = catnip(path1, 
                   path2 + prefix_mol + '/' + prefix_mol, 
                   path3 + prefix_pair + '/' + prefix_pair)
        print(prefix_mol, prefix_pair, j)
    else:
        prefix = 'dj-{}-{}{}{}' .format(int(delta * 1000), 
                                        ia,
                                        'xyz'[iv],
                                        ' +-'[sign])
        j = catnip(path1 + prefix + '/' + prefix, 
                   path2, 
                   path3 + prefix + '/' + prefix)
        print(prefix, j)
    return j
        
def multi_finite_dif(delta=0.01):
    from multiprocessing import Pool
    from functools import partial

    atoms = read('static.xyz')
    command = partial(finite_dif, delta, atoms)

    disps = []
    for ia, iv, sign in get_displacements(atoms):
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
                multi_finite_dif(delta)

    # run Gaussian for displacements of second molecule
    mol2 = str(pair[1][1] + 1)
    with chdir(mol2):
        if not os.path.isdir('displacements'):
            mkdir('displacements')
            with chdir('displacements'):
                copyfile('../' + mol2 + '.xyz', 'static.xyz')
                multi_finite_dif(delta)

    # run Gaussian for displacements of the pair
    molpair = str(pair[0])
    with chdir(molpair):
        mkdir('displacements')
        with chdir('displacements'):
            copyfile('../' + molpair + '.xyz', 'static.xyz')
            multi_finite_dif(delta)

def read_jdelta(delta=0.01, phonon_file='mesh.yaml', temp=0.025):
    from multiprocessing import Pool
    from functools import partial   
    
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)

    for pair in pairs.items():
        mol1 = str(int(pair[1][0]) + 1)
        mol2 = str(int(pair[1][1]) + 1)
        molpair = pair[0]
        if not os.path.exists(molpair + '_disp_js.npz'):
            # considering displacements of the first molecule
            path1 = mol1 + '/displacements/'
            path2 = mol2 + '/' + mol2
            path3 = molpair + '/displacements/'    

            disps = []
            atoms = read(path1 + 'static.xyz')
            offset = int(len(atoms))
            for ia, iv, sign in get_displacements(atoms):
                disps.append((ia, iv, sign))

            command = partial(read_finite_dif, delta, path1, path2, path3, None)
            with Pool(processes=64) as pool:
                jlists = pool.map(command, disps)            

            # considering displacements of the second molecule
            path1 = mol1 + '/' + mol1
            path2 = mol2 + '/displacements/'
            path3 = molpair + '/displacements/'    

            disps = []
            atoms = read(path2 + 'static.xyz')
            for ia, iv, sign in get_displacements(atoms):
                disps.append((ia, iv, sign))

            command = partial(read_finite_dif, delta, path1, path2, path3, offset)
            with Pool(processes=64) as pool:
                jlists += pool.map(command, disps)        

            data = {'js': jlists}   
            np.savez_compressed(molpair + '_disp_js.npz', data)

        else:
            jlists = np.load(molpair + '_disp_js.npz')['js']

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