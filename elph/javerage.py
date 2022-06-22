from ase.io import read
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy import sparse
import numpy as np
import os
from ase import Atoms
from ase.calculators.gaussian import Gaussian
from elph import chdir, mkdir
import subprocess
from os.path import exists
import json
from collections import OrderedDict


def find_structure_file(folder):
    """Searches the current working directory for the molecular structure file. 
        Allowed file types are .cif, .gen, .sdf, or .xyz. 

    Args:
        folder (str): The current working directory.

    Returns:
        str: Molecular structure file. 
    """
    import glob
    
    structure_file = glob.glob(folder + '/*.cif') + \
          glob.glob(folder + '/*.gen') + \
          glob.glob(folder + '/*.sdf') + \
          glob.glob(folder + '/*.xyz')
    print(structure_file)
    structure_file = structure_file[0]

    return structure_file

def get_centers_of_mass(atoms, n_components, component_list):
    """Gets centers of mass for each molecule

    Args:
        atoms (Atoms): Atoms in the system
        n_components (int): Number of molecules in the system
        component_list (dict): maps each atom number to the molecule it belongs to

    Returns:
        list: A list containing the centers of mass for each molecule in the system
    """
    centers_of_mass = []
    for i in range(n_components):
        molIdx_i = i
        molIdxs_i = [ x for x in range(len(component_list)) if component_list[x] == molIdx_i ]
        centers_of_mass.append(atoms[molIdxs_i].get_center_of_mass())
    return centers_of_mass

def compute_total_weight(centers_of_mass):
    """Computes the distances between the centers of mass. We represent the centers of mass as a fully-connected graph with the weight of each edge representing the distance between molecules

    Args:
        centers_of_mass (list): A list containing the centers of mass of each molecule in the system

    Returns:
        tuple: total_weight is a float containing the sum of all weights in the graph. has_dupes is a bool indicating whether there are two molecules overlapping.
    """
    total_weight = 0
    has_dupes = False
    for i in range(len(centers_of_mass)):
        for j in range(i+1, len(centers_of_mass)):
            total_weight +=  np.linalg.norm(centers_of_mass[i] - centers_of_mass[j])
            if np.linalg.norm(centers_of_mass[i] - centers_of_mass[j]) < 0.1:
                has_dupes = True
    return total_weight, has_dupes

def write_structure(label, component_list, molecules, all_atoms):
    """Writes the structure to a file

    Args:
        label (str): The label to be written. Pairs are A, B, C and molecules are 1, 2, 3
        component_list (dict): maps the atom number to the molecule it belongs to
        molecules (list): A list of the molecule numbers to be written
        all_atoms (Atoms): The Atoms object containing the atoms to be written
    """
    atoms = Atoms()
    if isinstance(molecules, list):
        for molecule in molecules:
            idxs = [ i for i in range(len(component_list)) if component_list[i] == molecule ]
            atoms.extend(all_atoms[idxs])
    else:
        idxs = [ i for i in range(len(component_list)) if component_list[i] == molecules ]
        atoms.extend(all_atoms[idxs])
    atoms.set_pbc([False, False, False])
    atoms.set_cell([0, 0, 0])

    mkdir(label)
    atoms.write(label + '/' + label + '.xyz')

def find_neighbors(atoms):
    """Identifies molecules by finding neighboring atoms

    Args:
        atoms (Atoms): Atoms in the system

    Returns:
        tuple: n_components is the number of molecules, component_list is a dict mapping atom number to molecule number, and edges are the a list containing which atoms are neighbors
    """
    neighbor_list = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix(neighbor_list.nl)
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    edges = list(matrix.keys())

    return n_components, component_list, edges

def is_triangle(centers_of_mass):
    l1 = np.linalg.norm(centers_of_mass[1] - centers_of_mass[0])
    l2 = np.linalg.norm(centers_of_mass[2] - centers_of_mass[0])
    l3 = np.linalg.norm(centers_of_mass[2] - centers_of_mass[1])
    return (l1 < l2 + l3) and (l2 < l1 + l3) and (l3 < l1 + l2)


def unwrap_atoms(structure_file=None):
    """Unwraps the molecules and identifies pairs based on a structure file.

    Args:
        structure_file (str, optional): Path to structure file. If none is given, local directory will be automatically searched to find a structure file. Defaults to None.

    Returns:
        str: json string containing the pair definitions based on molecule number.
    """
    folder = os.getcwd()

    if structure_file:
        if os.path.exists(folder + '/../' + structure_file):
            structure_file = folder + '/../' + structure_file
        else:
            structure_file = folder + '/' + structure_file
    else:
        structure_file = find_structure_file(folder)

    atoms = read(structure_file)
    neighbor_list = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix(neighbor_list.nl)
    n_components, component_list = sparse.csgraph.connected_components(matrix)

    idx = 0
    molIdx = component_list[idx]
    print("There are {} molecules in the system".format(n_components))
    molIdxs = [ i for i in range(len(component_list)) if component_list[i] == molIdx ]
    edges = list(matrix.keys())
    max_bond_len = max(natural_cutoffs(atoms))
    cell = list(atoms.get_cell())
    cell_array = atoms.get_cell()
    atoms.set_pbc([False,False,False])

    all_positions = atoms.get_positions()
    is_optimized = False
    # For each bond, take the lower left and move it upper right until the bond shrinks
    iterations = 0
    print("optimizing atoms")
    while not is_optimized:
        iterations += 1
        print("{} iterations".format(iterations))
        is_optimized = True
        for i in range(3):
            for edge in edges:
                positions = all_positions
                distance = np.linalg.norm(positions[edge[0]]-positions[edge[1]])
                if distance > max_bond_len*2:
                    min_pos = positions[edge[0]] if positions[edge[0],i] < positions[edge[1],i] else positions[edge[1]]
                    max_pos = positions[edge[0]] if positions[edge[0],i] >= positions[edge[1],i] else positions[edge[1]]
                    new_pos = min_pos
                    if np.linalg.norm(max_pos - (min_pos + cell[i])) < distance:
                        new_pos = min_pos + cell[i]
                        is_optimized = False
                    if np.array_equal(min_pos,positions[edge[0]]):
                        all_positions[edge[0]] = new_pos
                        all_positions[edge[1]] = max_pos
                    else:
                        all_positions[edge[0]] = max_pos
                        all_positions[edge[1]] = new_pos
    atoms.set_positions(all_positions)
    
    # Construct graph (compute total weight)
    original_centers_of_mass = get_centers_of_mass(atoms, n_components, component_list)
    centers_of_mass = np.copy(original_centers_of_mass)
    weight, has_dupes = compute_total_weight(centers_of_mass)
    test_dirs = cell
    test_dirs.extend([-x for x in test_dirs])

    is_optimized = False
    while not is_optimized:
        is_optimized = True
        for i in range(n_components):
            for j in range(6):
                test_centers_of_mass = np.copy(centers_of_mass)
                test_centers_of_mass[i] += test_dirs[j]
                test_weight, has_dupes = compute_total_weight(test_centers_of_mass)
                if test_weight < weight and not has_dupes:
                    centers_of_mass = np.copy(test_centers_of_mass)
                    weight = test_weight
                    is_optimized = False
    
    # Write centers of mass to file
    translations = np.zeros([len(atoms), 3])
    
    for i in range(n_components):
        molIdx = component_list[i]
        molIdxs = [ x for x in range(len(component_list)) if component_list[x] == molIdx ]
        
        dif = centers_of_mass[i] - original_centers_of_mass[i]
        translations[molIdxs,:] = dif

    atoms.translate(translations)
    atoms.center()

    original_components = n_components
    atoms = atoms * [2,1,1]
    n_components, component_list, edges = find_neighbors(atoms)


    new_atoms = Atoms()
    new_atoms.set_cell(atoms.get_cell())
    counter = 0
    atom_mapping = {}
    for i in range(3):
        if i == 2 and original_components >= 3:
            idx = original_components
        else:
            idx = i
        molIdxs = [ j for j in range(len(component_list)) if component_list[j] == idx ]
        new_atoms.extend(atoms[molIdxs])
        for idx in molIdxs:
            atom_mapping[idx] = counter 
            counter += 1


    with open('atom_mapping.json', 'w') as f:
        f.write(json.dumps(OrderedDict(sorted(atom_mapping.items(), key=lambda t: t[1])), indent=2))

    n_components, component_list, edges = find_neighbors(new_atoms)
    
    coms = get_centers_of_mass(new_atoms, n_components, component_list)
    weight, has_dupes = compute_total_weight(coms)

    states = [-1,0,1]

    best_weight = weight
    best_coms = np.copy(coms)
    same_state_count = 0
    while same_state_count < 3:
        same_state_count += 1
        for molecule_num in range(3):
            for i in states:
                for j in states:
                    for k in states:
                        vec = i*test_dirs[0] + j*test_dirs[1] + k*test_dirs[2]
                        test_coms = np.copy(coms)
                        test_coms[molecule_num] += vec
                        weight, has_dupes = compute_total_weight(test_coms)
                        if not has_dupes and is_triangle(test_coms):
                            if weight < best_weight:
                                best_weight = weight
                                best_coms = np.copy(test_coms)
                                same_state_count = 0
    translations = np.zeros([len(new_atoms), 3])
    displacements = best_coms - coms
    for i in range(3):
        molIdxs = [ j for j in range(len(component_list)) if component_list[j] == i ]
        translations[molIdxs] = displacements[i]
    new_atoms.translate(translations)

    final_atoms = Atoms()
    
    for i in range(3):
        molIdxs = [ j for j in range(len(component_list)) if component_list[j] == i ]
        final_atoms.extend(new_atoms[molIdxs])
        
    final_atoms.center()
    final_atoms.set_pbc([False, False, False])
    final_atoms.set_cell([0, 0, 0])
    n_components, component_list, edges = find_neighbors(final_atoms)
    final_atoms.write('all_pairs.xyz')
   
        
    # Create structures with each pair of atoms
    """
    Directory structure will be as follows:
    >main_folder
        >1
            -1.com
            ~1.log
            ~fort.7 > 1.pun
            >Displacements
                -0
                -1
                ...
        >2
            -2.com
            ~2.log
            ~fort.7 > 2.pun
        >3
            -3.com
            ~3.log
            ~fort.7 > 3.pun
        >A
            -PairA.com
            ~PairA.log
            ~fort.7 > PairA.pun
        >B
            -PairB.com
            ~PairB.log
            ~fort.7 > PairB.pun
        >C
            -PairC.com
            ~PairC.log
            ~fort.7 > PairC.pun
    """

    molecules = {'1':0, 
                 '2':1, 
                 '3':2}

    for key, value in molecules.items():
        write_structure(key, component_list, value, final_atoms)

    pairs = {'A':[0, 1], 
             'B':[1, 2],
             'C':[0, 2]}

    for key, value in pairs.items():
        cycle = value
        write_structure(key, component_list, cycle, final_atoms)

    with open('all_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)

    return list(molecules) + list(pairs)

def nersc_bash(name):
    """Write bash file and submit job

    Args:
        name (str): name of the job
    """
    cmd = os.environ['ASE_GAUSSIAN_COMMAND']
    with open('run.py', 'w') as f:
        f.write('#!/bin/bash \n'
                '#SBATCH -J {} \n'
                '#SBATCH -q flex \n'
                '#SBATCH -N 1 \n'
                '#SBATCH -t 03:00:00 \n'
                '#SBATCH --time-min 00:30:00 \n'
                '#SBATCH -C knl \n'
                '#SBATCH --output=out.out \n'
                '#SBATCH --error=err.out \n'
                '\n'
                'srun {} < {}.com > {}.log\n'
                'mv fort.7 {}.pun'
                .format(name, cmd, name, name, name))
    subprocess.run(['sbatch', 'run.py'])

def get_orbitals(atoms, name):
    """Runs gaussian to compute the orbitals for a system

    Args:
        atoms (Atoms): Atoms in the system
        name (str): The name of the system
    """
    if 'GAUSSIAN_CORES' in os.environ:
        c = os.environ['GAUSSIAN_CORES']
    else:
        c = 12
    if 'GAUSSIAN_BASIS' in os.environ:
        b = os.environ['GAUSSIAN_BASIS']
    else:
        b = '3-21G*'
    if not exists(name + '.pun'):
        calculator = Gaussian(mem='4GB',
                              nprocshared=c,
                              label=name,
                              save=None,
                              method='b3lyp',
                              basis=b,
                              scf='tight',
                              pop='full',
                              extra='nosymm punch=mo iop(3/33=1)')
        print("Running Gaussian calculation for ", name)
        calculator.write_input(atoms)
        nersc_bash(name)
    else:
        print(['Simulation {} is done' .format(name)])

def catnip(paths):
    """Runs Catnip to determine the transfer integral

    Args:
        paths (str): Paths to the first, second and pair Gaussian Result

    Returns:
        str: the transfer integral for the system
    """
    path1 = paths[0]
    path2 = paths[1]
    path3 = paths[2]
    path1 += '.pun'
    path2 += '.pun'
    path3 += '.pun'
    cmd = os.environ['ELPH_CATNIP_CMD']
    cmd += " -p_1 {} -p_2 {} -p_P {}" .format(path1, path2, path3)
    output = subprocess.check_output(cmd, shell=True)
    return output.decode('ascii').split()[-13]

def javerage():
    """Computes the transfer integral for all types of molecules (single or pairs)
    """
    folders = unwrap_atoms()

    for name in folders:
        with chdir(name):
            atoms = read(name + '.xyz')
            get_orbitals(atoms, name)

def read_javerage():
    """Reads the transfer integral for all pairs of molecules
    """
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)

    pp = []
    p1 = []
    p2 = []
    for pair in pairs.items():
        pp.append(pair[0] + '/' + pair[0])
        p1.append(str(int(pair[1][0]) + 1) + '/' + str(int(pair[1][0]) + 1))
        p2.append(str(int(pair[1][1]) + 1) + '/' + str(int(pair[1][1]) + 1))

    print("Calculating transfer integrals")
    from multiprocessing import Pool
    with Pool(processes=3) as pool:
        j = pool.map(catnip, zip(p1, p2, pp))

    for p, jj in zip(pairs.keys(), j):
        data = {p: jj}
        with open('J_' + p + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    javerage()