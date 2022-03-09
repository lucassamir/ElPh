import numpy as np
from ase.calculators.gaussian import Gaussian
from ase.io import read
from os.path import exists
from shutil import copyfile
from tloc.javerage import get_orbitals
from tloc import chdir, mkdir

def get_all_displacements(atoms):
    for ia in range(len(atoms)):
        for iv in range(3):
            for sign in [-1, 1]:
                yield (ia, iv, sign)

def displace_atom(atoms, ia, iv, sign, delta):
    new_atoms = atoms.copy()
    pos_av = new_atoms.get_positions()
    pos_av[ia, iv] += sign * delta
    new_atoms.set_positions(pos_av)
    return new_atoms

def finite_dif(delta=0.01):
    atoms = read('static.xyz')
    for ia, iv, sign in get_all_displacements(atoms):
        prefix = 'dj-{}-{}{}{}' .format(int(delta * 1000), 
                                        ia,
                                        'xyz'[iv],
                                        ' +-'[sign])
        if not exists(prefix + '.pun'):
            new_structure = displace_atom(atoms, ia, iv, sign, delta)
            get_orbitals(new_structure, prefix)

def derivative(ia, iv, delta1, delta2):
    dj = []
    for delta in [delta1, delta2]:
        for sign in [-1, 1]:
            prefix = 'dj-{}-{}{}{}' .format(int(delta * 1000), 
                                            ia,
                                            'xyz'[iv],
                                            ' +-'[sign])
            with open(prefix + '.txt') as f:
                dj.append(f.read())
    return (-dj[3] + 8 * dj[1] - 8 * dj[0] + dj[2]) / (12 * delta1)

def get_dj_matrix(delta1, delta2):
    atoms = read('static.xyz')
    dj_ik = np.zeros([len(atoms), 3])
    for ia in range(len(atoms)):
        for iv in range(3):
            dj_ik[ia, iv] = derivative(ia, iv, delta1, delta2)

def get_jdelta(pair, delta=0.01):
    # run Gaussian for displacements of first molecule
    name = str(pair[1][0] + 1)
    with chdir(name):
        mkdir('displacements')
        with chdir('displacements'):
            copyfile('../' + name + '.xyz', 'static.xyz')
            finite_dif(delta / 2)
            finite_dif(delta)

    # run Gaussian for displacements of first molecule in the pair
    name = str(pair[0])
    with chdir(name):
        mkdir('displacements')
        with chdir('displacements'):
            copyfile('../' + name + '.xyz', 'static.xyz')
            finite_dif(delta / 2)
            finite_dif(delta)

if __name__ == '__main__':
    pairs = {'A':[0, 1], 
             'B':[1, 2],
             'C':[0, 2]}
    
    for pair in pairs.items():
        get_jdelta(pair, delta=0.01)
