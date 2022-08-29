import numpy as np
import json
from elph.sigma import load_phonons, get_dj_matrix
import cmcrameri.cm as cmc

def heat_grad(molpair, dj_matrix_av):
    from ase.io import read
    import matplotlib.pyplot as plt
    from ase.data import covalent_radii

    atoms = read(molpair + '/' + molpair + '.xyz')
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    radii = [covalent_radii[num]*300 for num in numbers]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    nmax = max(max(x), max(y), max(z))
    nmin = min(min(x), min(y), min(z))

    data = {'x': x,
            'y': y,
            'z': z,
            'size': radii}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    u = dj_matrix_av[:, 0] * 35
    v = dj_matrix_av[:, 1] * 35
    w = dj_matrix_av[:, 2] * 35
    data['u'] = u
    data['v'] = v
    data['w'] = w
    ax.quiver(x, y, z, u, v, w, color='black')
    ax.scatter(x, y, z, s = radii, alpha=1, edgecolors='grey', c='lavender')
    ax.set_xlim3d(nmin, nmax)
    ax.set_ylim3d(nmin, nmax)
    ax.set_zlim3d(nmin, nmax)
    plt.axis('off')
    plt.show()

    np.savez_compressed('view_grad_' + molpair + '.npz', **data)


def heat_atoms(molpair, ssigma_eav):
    """Scatter plot of atoms with heat indicating 
    the highest contributions to sigma. Save plot data 
    (atomic positions, atomic sizes and sigma contribution) 
    to numpy compressed file.

    Args:
        molpair (str): Molecular pair label
        ssigma_eav (array): Squared sigma array of sizes modes, atoms and directions
    """
    from ase.io import read
    from ase.data import covalent_radii
    import matplotlib.pyplot as plt
    from matplotlib import cm
    # import plotly.graph_objects as go

    atoms = read(molpair + '/' + molpair + '.xyz')
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    radii = [covalent_radii[num]*300 for num in numbers]    
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ssigma = np.sum(np.sum(ssigma_eav, axis=-1), axis=0)
    nmax = max(max(x), max(y), max(z))
    nmin = min(min(x), min(y), min(z))

    with open('J_' + molpair + '.json', 'r') as json_file:
        j = np.abs(np.float(json.load(json_file)[molpair]))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, s = radii, c=ssigma*10**5, vmin=0, vmax=1, cmap=cmc.devon_r, alpha=1, edgecolors='grey')
    ax.set_xlim3d(nmin, nmax)
    ax.set_ylim3d(nmin, nmax)
    ax.set_zlim3d(nmin, nmax)
    fig.colorbar(s)
    plt.axis('off')
    plt.show()

    data = {'x': x,
            'y': y,
            'z': z,
            'size': radii,
            'sigma': ssigma / j**2}

    np.savez_compressed('view_atoms_' + molpair + '.npz', **data)

    # layout = go.Layout(
    #          scene=dict(
    #              aspectmode='data'
    #         ))
    # fig = go.Figure(data=[go.Scatter3d(x=x, 
    #                                    y=y, 
    #                                    z=z, 
    #                                    mode='markers',
    #                                    marker=dict(size=10*np.sqrt(masses/np.pi),
    #                                                color=sigma,
    #                                                colorscale='Viridis',
    #                                                opacity=0.8,
    #                                                ))],
    #                 layout=layout)
    # fig.write_html(molpair + '.html')
    # fig.show()

def heat_modes(molpair, ssigma_eav, vecs_eav, n):
    """Scatter plot of atoms with arrows indicating
    phonon displacements of n phonons modes with the 
    highest sigma contributions. Save plot data 
    (atomic positions, vector directions, atomic sizes 
    and sigma contribution) to numpy compressed file.

    Args:
        molpair (str): Molecular pair label
        ssigma_eav (array): Squared sigma array of sizes modes, atoms and directions
        vecs_eav (array): Array of phonon eigendisplacements of size modes, atoms and directions
        n (int): Top n phonon modes with highest sigma contribution
    """
    from ase.io import read
    from ase.data import covalent_radii
    import matplotlib.pyplot as plt

    atoms = read(molpair + '/' + molpair + '.xyz')
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    radii = [covalent_radii[num]*300 for num in numbers]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    # ssigma = np.sum(np.sum(ssigma_eav, axis=-1), axis=-1)
    ssigma = ssigma_eav[::36]
    nmax = max(max(x), max(y), max(z))
    nmin = min(min(x), min(y), min(z))

    ind = np.argsort(ssigma)[-n:][::-1]

    data = {'x': x,
            'y': y,
            'z': z,
            'size': radii}
    data_a = np.load('ratio_A.npz')
    freqs = data_a['freqs']* 8065

    for i in ind:
        print(len(freqs))
        print(freqs[i])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        u = vecs_eav[i, :, 0] * 200
        v = vecs_eav[i, :, 1] * 200
        w = vecs_eav[i, :, 2] * 200
        data['u'] = u
        data['v'] = v
        data['w'] = w
        ax.scatter(x, y, z, s = radii, alpha=1, edgecolors='grey', c='lavender')
        ax.quiver(x, y, z, u, v, w, color='black')
        ax.set_xlim3d(nmin, nmax)
        ax.set_ylim3d(nmin, nmax)
        ax.set_zlim3d(nmin, nmax)
        plt.axis('off')
        plt.show()

        np.savez_compressed('view_modes_' + molpair + '_' + '{}' .format(i) + '_' + '.npz', **data)

def sigma_contribution(pair_atoms, dj_av, temp):
    """Standard deviation (sigma) in function of
    modes, atoms and directions.

    Args:
        pair_atoms (list): List containing atom indices of molecular pair
        dj_av (array): Delta of J in function of atoms and directions
        temp (float): Temperature in eV

    Returns:
        typle: sigma and eigendisplacements in function of modes, atoms and directions
    """
    freqs_e, vecs_eav, nq = load_phonons(pair_atoms)
    epcoup_eav = vecs_eav * dj_av[None, :, :]
    ssigma_eav = (1 / nq) * (epcoup_eav**2 / (2 * np.tanh(freqs_e[:, None, None] / (2 * temp))))
    epcoup_e = np.einsum('av,eav->e', dj_av, vecs_eav)
    ssigma_e = (1 / nq) * epcoup_e**2 / \
        (2 * np.tanh(freqs_e / (2 * temp)))

    return ssigma_e, vecs_eav

def get_sigma(pair, delta, temp, mode, n):
    """Read transfer integrals of displacements systems and calculate sigma

    Args:
        pair (dic): Dictionary containing molecular indices of pair
        delta (float): Step size to displace atoms in Ang
        temp (float): Temperature in eV
        mode (str): Mode of visualization (atoms or modes)
        n (int): Top n phonon modes with highest sigma contribution

    Raises:
        NotImplementedError: Error raised if the chosen mode is not atoms or modes
    """
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

    ssigma_eav, vecs_eav = sigma_contribution(pair_atoms, dj_matrix_av, temp)

    if mode == 'atoms':
        heat_atoms(molpair, ssigma_eav)
    elif mode == 'modes':
        heat_modes(molpair, ssigma_eav, vecs_eav, n)
    elif mode == 'gradient':
        heat_grad(molpair, dj_matrix_av)
    else:
        msg = 'The available visualization results are gradient, atoms and modes'
        raise NotImplementedError(msg)

def view(mode='atoms', n=3):
    """Main function of the visualization module. There are three modes:

    [gradient]: Scatter plot of atoms with arrows indicating 
    the gradientof the transfer integral.

    [atoms]: Scatter plot of atoms with heat indicating 
    the highest contributions to sigma.

    [modes] [[n]] : Scatter plot of atoms with arrows indicating
    phonon displacements of n phonons modes with the 
    highest sigma contributions

    Args:
        mode (str, optional): Mode of visualization (atoms or modes). Defaults to 'atoms'.
        n (int, optional): Top n phonon modes with highest sigma contribution. Only useful for modes. Defaults to 3.
    """
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)
    
    # for pair in pairs.items():
    #     get_sigma(pair, delta=0.01, temp=0.025, mode=mode, n=int(n))
    pair = ('A', [0, 1])
    get_sigma(pair, delta=0.01, temp=0.025, mode=mode, n=int(n))

if __name__ == '__main__':
    view()