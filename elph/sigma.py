import numpy as np
from ase.io import read
from shutil import copyfile
from elph.javerage import get_orbitals, catnip
from elph import chdir, mkdir
import json
import os

def load_phonons(pair_atoms, phonon_file='phonon.npz', map_file='atom_mapping.json'):
    """Loads phonon modes and returns frequencies, eigenvectors and number of q points

    Args:
        pair_atoms (list): Atomic indices of the interacting pair
        phonon_file (str, optional): Numpy file with the phonon modes. Defaults to 'phonon.npz'
        map_file (str, optional): JSON file generate to map atomic indices 
        of the wraped to the unwraped structure. Defaults to 'atom_mapping.json'

    Returns:
        tuple: vector of frequencies, vector of eigenvectors, and integer of the number of qpoints
    """
    # read phonon modes file
    phonon = np.load(phonon_file)

    # read mapping file
    with open(map_file, 'r') as json_file:
        mapping = list(map(int, json.load(json_file).keys()))
    
    # use mapping to order the wrapped phonon modes
    # based on the unwrapped atoms
    vecs_eav = np.tile(phonon['vecs'], [1, 2, 1])[:, mapping, :]

    # selecting only the phonon modes relevant to the 
    # interaction pair of molecules
    vecs_eav = vecs_eav[:, pair_atoms, :]

    return phonon['freqs'], vecs_eav, phonon['nq']

def get_dj_matrix(jlists, delta):
    """Matrix containing gradient of J, the transfer integral

    Args:
        jlists (list): list of transfer integrals
        delta (float): size of displacement

    Returns:
        array: matrix containing gradient of j
    """
    latoms = len(jlists) // 6

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

    dj_matrix = (np.abs(jp) - np.abs(jm)) / (2 * delta)

    return dj_matrix

def get_deviation(pair_atoms, dj_av, temp):
    """Calculate standard deviation (sigma) of the transfer integral

    Args:
        pair_atoms (array): Numpy array of atomic indices of interacting pairs
        dj_av (array): Numpy array of gradient of J for all atoms in the pair for all directions
        temp (float): Temperature in eV

    Returns:
        float: standard deviation (sigma) of the transfer integral
    """
    freqs_e, vecs_eav, nq = load_phonons(pair_atoms)
    epcoup_e = np.einsum('av,eav->e', dj_av, vecs_eav)
    ssigma = (1 / nq) * np.sum(epcoup_e**2 / \
        (2 * np.tanh(freqs_e / (2 * temp))))

    return np.sqrt(ssigma)

def get_displacements(atoms):
    """Returns displacement of each atom in each direction

    Args:
        atoms (Atoms): Atoms objects

    Yields:
        yield: (atom number, direction, sign)
    """
    latoms = len(atoms)
    for ia in range(latoms):
        for iv in range(3):
            for sign in [-1, 1]:
                yield (ia, iv, sign)

def displace_atom(atoms, ia, iv, sign, delta):
    """Displace one atomic position in the Atoms object

    Args:
        atoms (Atoms): Atoms object
        ia (int): atom index
        iv (int): direction index
        sign (str): sign of displacement in each direction
        delta (float): size of displacement

    Returns:
        Atoms: Updated Atoms object
    """
    new_atoms = atoms.copy()
    pos_av = new_atoms.get_positions()
    pos_av[ia, iv] += sign * delta
    new_atoms.set_positions(pos_av)
    return new_atoms

def finite_dif(delta, atoms, disp):
    """Compute Gaussian calculation for displaced system

    Args:
        delta (float): Size of displacement
        atoms (Atoms): Atoms object
        disp (tuple): (atom index, direction and sign)
    """
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
    """Calculate transfer integral for displaced system

    Args:
        delta (float): Size of displacement
        path1 (str): Path of first molecule
        path2 (str): Path of second molecule
        path3 (str): Path of pair of molecules
        offset (int): Number of atoms in the first molecule
        disp (tuple): (atom index, direction and sign)

    Returns:
        float: Calculated transfer integral
    """
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
        j = catnip([path1, 
                    path2 + prefix_mol + '/' + prefix_mol, 
                    path3 + prefix_pair + '/' + prefix_pair])
        print(prefix_pair, j)
    else:
        prefix = 'dj-{}-{}{}{}' .format(int(delta * 1000), 
                                        ia,
                                        'xyz'[iv],
                                        ' +-'[sign])
        j = catnip([path1 + prefix + '/' + prefix, 
                    path2, 
                    path3 + prefix + '/' + prefix])
        print(prefix, j)
    return j
        
def multi_finite_dif(delta=0.01):
    """Multiprocessing to submit jobs in parallel

    Args:
        delta (float, optional): Size of displacement. Defaults to 0.01.
    """
    from multiprocessing import Pool
    from functools import partial

    atoms = read('static.xyz')
    command = partial(finite_dif, delta, atoms)

    disps = []
    for ia, iv, sign in get_displacements(atoms):
        disps.append((ia, iv, sign))

    with Pool(processes=64) as pool:
        pool.map(command, disps)

def run_sigma(pair, delta=0.01):
    """Using the finite differences methods, 
    create folder for each displace atom,
    and calculate transfer integral.

    Args:
        pair (tuple): Pair name and molecules indices
        delta (float): Size of displacement. Defaults to 0.01.
    """
    print('Running Gaussian for displacements')

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
        if not os.path.isdir('displacements'):
            mkdir('displacements')
            with chdir('displacements'):
                copyfile('../' + molpair + '.xyz', 'static.xyz')
                multi_finite_dif(delta)

def read_sigma(delta=0.01, temp=0.025):
    """Write phonon modes from phonopy result, 
    read transfer integrals of finite differences of one pair,
    and calculate standard deviation (sigma) of all pairs

    Args:
        delta (float, optional): Size of displacement. Defaults to 0.01.
        temp (float, optional): Temperature in eV. Defaults to 0.025.
    """
    from multiprocessing import Pool
    from functools import partial   
    from elph.phonons import write_phonons

    write_phonons()
    
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)

    for pair in pairs.items():
        mol1 = str(int(pair[1][0]) + 1)
        mol2 = str(int(pair[1][1]) + 1)
        molpair = pair[0]
        if not os.path.exists(molpair + '_disp_js.npz'):
            print("Calculating transfer integrals")

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
            np.savez_compressed(molpair + '_disp_js.npz', **data)

        else:
            jlists = np.load(molpair + '_disp_js.npz')['js']

        # Create GradJ matrix with a atoms and v directions
        dj_matrix_av = get_dj_matrix(jlists, delta)

        # Calculate sigma
        offset = len(dj_matrix_av) // 2
        pair_atoms = np.concatenate([np.arange((int(mol1) - 1) * offset, 
                                                int(mol1) * offset), 
                                     np.arange((int(mol2) - 1) * offset, 
                                                int(mol2) * offset)])
        
        sigma = get_deviation(pair_atoms, dj_matrix_av, temp)
        data = {molpair: sigma}
        with open('Sigma_' + molpair + '.json', 'w', encoding='utf-8') as f:
             json.dump(data, f, ensure_ascii=False, indent=4)

def sigma():
    """Calculate the standard deviation (sigma)
    for each pair of molecules. Write sigmas to JSON files
    """
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)
    
    for pair in pairs.items():
        run_sigma(pair, delta=0.01)

if __name__ == '__main__':
    sigma()