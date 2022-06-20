import numpy as np
import json
from tloc.sigma import load_phonons, get_dj_matrix

def heat_atoms(molpair, sigma_eav):
    """Scatter plot of atoms with heat indicating 
    the highest contributions to sigma. Save plot data 
    (atomic positions, atomic sizes and sigma contribution) 
    to numpy compressed file.

    Args:
        molpair (str): Molecular pair label
        sigma_eav (array): Sigma array of sizes modes, atoms and directions
    """
    from ase.io import read
    import matplotlib.pyplot as plt
    # import plotly.graph_objects as go

    atoms = read(molpair + '/' + molpair + '.xyz')
    pos = atoms.get_positions()
    masses = atoms.get_masses()
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    sigma = np.sum(np.sum(sigma_eav, axis=-1), axis=0)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, s = 20 * masses, c=sigma)
    fig.colorbar(s)
    plt.show()

    data = {'x': x,
            'y': y,
            'z': z,
            'size': 20 * masses,
            'sigma': sigma}

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

def heat_modes(molpair, sigma_eav, vecs_eav, n):
    """Scatter plot of atoms with arrows indicating
    phonon displacements of n phonons modes with the 
    highest sigma contributions. Save plot data 
    (atomic positions, vector directions, atomic sizes 
    and sigma contribution) to numpy compressed file.

    Args:
        molpair (str): Molecular pair label
        sigma_eav (array): Sigma array of sizes modes, atoms and directions
        vecs_eav (array): Array of phonon eigendisplacements of size modes, atoms and directions
        n (int): Top n phonon modes with highest sigma contribution
    """
    from ase.io import read
    import matplotlib.pyplot as plt

    atoms = read(molpair + '/' + molpair + '.xyz')
    pos = atoms.get_positions()
    masses = atoms.get_masses()
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    sigma = np.sum(np.sum(sigma_eav, axis=-1), axis=-1)

    ind = np.argsort(sigma)[-n:][::-1]

    data = {'x': x,
            'y': y,
            'z': z,
            'size': 20 * masses}

    for i in ind:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        u = vecs_eav[i, :, 0]
        v = vecs_eav[i, :, 1]
        w = vecs_eav[i, :, 2]
        data['u'] = u
        data['v'] = v
        data['w'] = w
        ax.quiver(x, y, z, u, v, w, length=2, normalize=True)
        ax.scatter(x, y, z, s = 20 * masses)
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
    freqs_e, vecs_eav, _ = load_phonons(pair_atoms)
    epcoup_eav = vecs_eav * dj_av[None, :, :]
    ssigma_eav = epcoup_eav**2 / (2 * np.tanh(freqs_e[:, None, None] / (2 * temp)))

    return np.sqrt(ssigma_eav), vecs_eav

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

    sigma_eav, vecs_eav = sigma_contribution(pair_atoms, dj_matrix_av, temp)

    if mode == 'atoms':
        heat_atoms(molpair, sigma_eav)
    elif mode == 'modes':
        heat_modes(molpair, sigma_eav, vecs_eav, n)
    else:
        msg = 'The available visualization results are atoms and modes'
        raise NotImplementedError(msg)

def view(mode='atoms', n=3):
    """Main function of the visualization module. There are two modes:

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
    
    for pair in pairs.items():
        get_sigma(pair, delta=0.01, temp=0.025, mode=mode, n=int(n))

if __name__ == '__main__':
    view()