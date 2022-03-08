import numpy as np
from ase.calculators.gaussian import Gaussian
from ase.io import read
from shutil import move

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
    move('fort.7', name)

def derivative(i, k, delta1, delta2):
    dj = []
    for delta in [delta1, delta2]:
        for sign in [-1, 1]:
            prefix = 'dj-{}-{}{}{}' .format(delta, i,
                                            'xyz'[k],
                                            ' +-'[sign])
            with open(prefix + '.txt') as f:
                dj.append(f.read())
    return (-dj[3] + 8 * dj[1] - 8 * dj[0] + dj[2]) / (12 * delta1)

def get_dj_matrix(delta1, delta2):
    atoms = read('static.xyz')
    i = len(atoms)
    dj_ik = np.zeros([i, 3])
    for ii in range(i):
        for k in range(3):
            dj_ik[ii, k] = derivative(ii, k, delta1, delta2)

def finite_dif(delta=0.01):
    atoms = read('static.xyz')
    pos_ik = atoms.get_positions()
    indices = list(range(len(atoms)))

    for i in indices:
        for k in range(3):
            for sign in [-1, 1]:
                # Update atomic positions
                atoms.positions = pos_ik
                atoms.positions[i, k] = pos_ik[i, k] + sign * delta
                prefix = 'dj-{}-{}{}{}' .format(delta, i,
                                                'xyz'[k],
                                                ' +-'[sign])
                get_orbitals(atoms, prefix)

def jdelta(delta=0.01):
    d1 = delta / 2
    d2 = delta
    finite_dif(d1)
    finite_dif(d2)
    get_dj_matrix(d1, d2)

if __name__ == '__main__':
    jdelta(delta=0.01)
