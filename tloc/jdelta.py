import numpy as np
from ase.calculators.gaussian import Gaussian
from ase.io import read
from os import rename
from os.path import exists

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

def get_orbitals(atoms, name):
    atoms.calc = Gaussian(mem='4GB',
                          nprocshared=24,
                          label=name,
                          save=None,
                          method='b3lyp',
                          basis='6-31G*',
                          scf='tight',
                          pop='full',
                          extra='nosymm punch=mo iop(3/33=1)')
    atoms.get_potential_energy()
    rename('fort.7', name + '.pun')

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

def jdelta(delta=0.01):
    d1 = delta / 2
    d2 = delta
    finite_dif(d1)
    finite_dif(d2)
    get_dj_matrix(d1, d2)

if __name__ == '__main__':
    jdelta(delta=0.01)
