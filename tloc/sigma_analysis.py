from locale import normalize
from matplotlib.pyplot import tight_layout
import numpy as np
import json
from tloc.jdelta import load_phonons, get_dj_matrix

def heat_atoms(molpair, sigma_eav):
    from ase.io import read
    import matplotlib.pyplot as plt

    atoms = read(molpair + '.xyz')
    pos = atoms.get_positions()
    masses = atoms.get_masses()
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    sigma = np.sum(np.sum(sigma_eav, axis=-1), axis=0)
    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, s = 20 * masses, c=sigma)
    
    plt.show()

def heat_modes(molpair, sigma_eav, vecs_eav, n):
    from ase.io import read
    import matplotlib.pyplot as plt

    atoms = read(molpair + '.xyz')
    pos = atoms.get_positions()
    masses = atoms.get_masses()
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    sigma = np.sum(np.sum(sigma_eav, axis=-1), axis=-1)

    ind = np.argsort(sigma)[-n:][::-1]

    for i in ind:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        ax.quiver(x, y, z, vecs_eav[i, :, 0], vecs_eav[i, :, 1], vecs_eav[i, :, 2], length=2, normalize=True)
        ax.scatter(x, y, z, s = 20 * masses)
        plt.show()

def sigma_contribution(pair_atoms, dj_av, temp):
    freqs_e, vecs_eav, _ = load_phonons(pair_atoms)
    epcoup_eav = vecs_eav * dj_av[None, :, :]
    ssigma_eav = epcoup_eav**2 / (2 * np.tanh(freqs_e[:, None, None] / (2 * temp)))

    return np.sqrt(ssigma_eav), vecs_eav

def get_sigma(pair, delta, temp):
    mol1 = str(int(pair[1][0]) + 1)
    mol2 = str(int(pair[1][1]) + 1)
    molpair = pair[0]

    jlists = np.load(molpair + '_disp_js.npz')['js']
    dj_matrix_av = get_dj_matrix(jlists, delta)
    offset = len(dj_matrix_av) // 2
    pair_atoms = np.concatenate([np.arange((int(mol1) - 1) * offset, 
                                            int(mol1) * offset), 
                                 np.arange((int(mol2) - 1) * offset, 
                                            int(mol2) * offset)])

    sigma_eav, vecs_eav = sigma_contribution(pair_atoms, dj_matrix_av, temp)
    #heat_atoms(molpair, sigma_eav)
    heat_modes(molpair, sigma_eav, vecs_eav, 3)

def sigma():
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)
    
    for pair in pairs.items():
        get_sigma(pair, delta=0.01, temp=0.025)

if __name__ == '__main__':
    sigma()