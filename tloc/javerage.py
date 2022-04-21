from ase.io import read
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy import sparse
import numpy as np
import os
from ase import Atoms
from ase.calculators.gaussian import Gaussian
from tloc import chdir, mkdir
import subprocess
from os.path import exists
import json

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
    print("Reading structure ", structure_file)
    structure_file = structure_file[0]

    return structure_file

def get_centers_of_mass(atoms, n_components, component_list):
    centers_of_mass = []
    for i in range(n_components):
        molIdx_i = i
        molIdxs_i = [ x for x in range(len(component_list)) if component_list[x] == molIdx_i ]
        centers_of_mass.append(atoms[molIdxs_i].get_center_of_mass())
    return centers_of_mass

def write_structure(label, component_list, molecules, all_atoms):
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
    print("Identifying molecules")
    neighbor_list = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix(neighbor_list.nl)
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    edges = list(matrix.keys())

    return n_components, component_list, edges

def unwrap_atoms(structure_file=None):
    folder = os.getcwd()
    structure_file = find_structure_file(folder)
    atoms = read(structure_file)
    atoms *= [3, 3, 3]
    
    n_components, component_list, edges = find_neighbors(atoms)

    # Compute total weight of each molecule and record minimum weight
    positions = atoms.get_positions()
    weights = []
    min_weight = float('inf')
    print("Finding fully-connected molecules")
    for idx in range(n_components):
        weight = 0
        molIdxs = [ i for i in range(len(component_list)) if component_list[i] == idx ]
        for edge in edges:
            if edge[0] in molIdxs:
                weight += np.linalg.norm(positions[edge[0]]-positions[edge[1]])
        weights.append(weight)
        if weight < min_weight:
            min_weight = weight

    # Keep only atoms which belong to molecules which have minimum weight
    keep_idx = []
    for i, weight in enumerate(weights):
        if weight <= min_weight + 1:
            keep_idx.append(i)
    keep_idxs = [ i for i in range(len(component_list)) if component_list[i] in keep_idx ]
    fully_connected_atoms = atoms[keep_idxs]

    # Re-compute molecules so that they fall in order
    n_components, component_list, edges = find_neighbors(fully_connected_atoms)

    # Compute centers of mass for remaining molecules
    centers_of_mass = get_centers_of_mass(fully_connected_atoms, n_components, component_list)
    com_edges = {}
    print("Finding 3 closest fully-connected molecules")
    for i in range(n_components):
        node_1 = centers_of_mass[i]
        for j in range(i+1, n_components):
            node_2 = centers_of_mass[j]
            com_edges[(i,j)] = np.linalg.norm(node_1 - node_2)
            
    # Identify 3 nearest centers of mass to each other Floyd Warshall is best, but it is already N^3. Just do brute force
    min_cycle_length = float('inf')
    min_cycle = [None, None, None]
    for i in range(len(centers_of_mass)):
        for j in range(i+1, len(centers_of_mass)):
            for k in range(j+1, len(centers_of_mass)):
                # TODO: Impose triangle condition
                cycle_length = com_edges[(i,j)] + com_edges[(j,k)] + com_edges[(i,k)]
                is_triangle = com_edges[(i,j)] < com_edges[(j,k)] + com_edges[(i,k)] and \
                                com_edges[(j,k)] < com_edges[(i,k)] + com_edges[(i,j)] and \
                                com_edges[(i,k)] < com_edges[(i,j)] + com_edges[(j,k)]
                if cycle_length < min_cycle_length and is_triangle:
                    min_cycle_length = cycle_length
                    min_cycle = [i, j, k]

    # Keep only these atoms
    keep_idxs = [ i for i in range(len(component_list)) if component_list[i] in min_cycle ]
    new_atoms = Atoms()
    for idx in min_cycle:
        keep_idxs = [ i for i in range(len(component_list)) if component_list[i] == idx ]
        new_atoms.extend(fully_connected_atoms[keep_idxs])
    # Center in cell
    new_atoms.center()
    new_atoms.set_pbc([False, False, False])
    new_atoms.set_cell([0, 0, 0])
    new_atoms.write('all_pairs.xyz')

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
        write_structure(key, component_list, min_cycle[value], fully_connected_atoms)

    pairs = {'A':[0, 1], 
             'B':[1, 2],
             'C':[0, 2]}

    for key, value in pairs.items():
        cycle = [min_cycle[v] for v in value]
        write_structure(key, component_list, cycle, fully_connected_atoms)

    with open('all_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)

    return list(molecules) + list(pairs)

def nersc_bash(name):
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
    if not exists(name + '.pun'):
        calculator = Gaussian(mem='48GB',
                              nprocshared=64,
                              label=name,
                              save=None,
                              method='b3lyp',
                              basis='6-31G',
                              scf='tight',
                              pop='full',
                              extra='nosymm punch=mo iop(3/33=1)')
        print("Running Gaussian calculation for ", name)
        calculator.write_input(atoms)
        nersc_bash(name)
    else:
        print(['Simulation {} is done' .format(name)])

def catnip(paths):
    print("Calculating transfer integral")
    path1 = paths[0]
    path2 = paths[1]
    path3 = paths[2]
    path1 += '.pun'
    path2 += '.pun'
    path3 += '.pun'
    cmd = os.environ['TLOC_CATNIP_CMD']
    cmd += " -p_1 {} -p_2 {} -p_P {}" .format(path1, path2, path3)
    output = subprocess.check_output(cmd, shell=True)
    return output.decode('ascii').split()[-2]

def javerage():
    folders = unwrap_atoms()

    for name in folders:
        with chdir(name):
            atoms = read(name + '.xyz')
            get_orbitals(atoms, name)

def read_javerage():
    with open('all_pairs.json', 'r') as json_file:
        pairs = json.load(json_file)

    pp = []
    p1 = []
    p2 = []
    for pair in pairs.items():
        pp.append(pair[0] + '/' + pair[0])
        p1.append(str(int(pair[1][0]) + 1) + '/' + str(int(pair[1][0]) + 1))
        p2.append(str(int(pair[1][1]) + 1) + '/' + str(int(pair[1][1]) + 1))

    from multiprocessing import Pool
    with Pool(processes=3) as pool:
        j = pool.map(catnip, zip(p1, p2, pp))

    for p, jj in zip(pairs.keys(), j):
        data = {p: jj}
        with open('J_' + p + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    javerage()