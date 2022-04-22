from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy import sparse
import numpy as np
import os
from tqdm.auto import trange, tqdm
from halo import Halo
from tloc import chdir, mkdir
from ase.io.trajectory import Trajectory
from ase import Atoms
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
    print(structure_file)
    structure_file = structure_file[0]

    return structure_file

def write_structure(label, component_list, molecules, all_atoms):
    atoms = Atoms()
    atom_mapping = {}
    counter = 0
    if isinstance(molecules, list):
        for molecule in molecules:
            idxs = [ i for i in range(len(component_list)) if component_list[i] == molecule ]
            for idx in idxs:
                if idx%140 in atom_mapping.keys():
                    atom_mapping[idx%140].append(counter)
                else:
                    atom_mapping[idx%140] = [counter] 
                counter += 1
            atoms.extend(all_atoms[idxs])
    else:
        idxs = [ i for i in range(len(component_list)) if component_list[i] == molecules ]
        for idx in idxs:
            if idx%70 in atom_mapping.keys():
                atom_mapping[idx%70].append(counter)
            else:
                atom_mapping[idx%70] = [counter] 
            counter += 1
        atoms.extend(all_atoms[idxs])
    atoms.set_pbc([False, False, False])
    atoms.set_cell([0, 0, 0])

    mkdir(label)
    atoms.write(label + '/' + label + '.xyz')
    with open(label + '/' + 'atom_mapping.json', 'w') as f:
        f.write(json.dumps(atom_mapping))

@Halo(text="Identifying molecules", color='green', spinner='dots')
def find_neighbors(atoms):
    neighbor_list = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix(neighbor_list.nl)
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    edges = list(matrix.keys())

    return n_components, component_list, edges

def compute_total_weight(centers_of_mass):
    
    total_weight = 0
    has_dupes = False
    for i in range(len(centers_of_mass)):
        for j in range(i+1, len(centers_of_mass)):
            total_weight +=  np.linalg.norm(centers_of_mass[i] - centers_of_mass[j])
            if np.linalg.norm(centers_of_mass[i] - centers_of_mass[j]) == 0:
                has_dupes = True
    return total_weight, has_dupes

def is_coplanar(centers_of_mass):
    a1 = centers_of_mass[2,0] - centers_of_mass[1,0]
    b1 = centers_of_mass[2,1] - centers_of_mass[1,1]
    c1 = centers_of_mass[2,2] - centers_of_mass[1,2]
    a2 = centers_of_mass[3,0] - centers_of_mass[1,0]
    b2 = centers_of_mass[3,1] - centers_of_mass[1,1]
    c2 = centers_of_mass[3,2] - centers_of_mass[1,2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * centers_of_mass[1,0] - b *centers_of_mass[1,1] - c * centers_of_mass[1,2])

    return abs(a * centers_of_mass[0,0] + b * centers_of_mass[0,1] + c * centers_of_mass[0,2] + d) <= 0.1

def get_centers_of_mass(atoms, n_components, component_list):
    centers_of_mass = []
    for i in range(n_components):
        molIdx_i = i
        molIdxs_i = [ x for x in range(len(component_list)) if component_list[x] == molIdx_i ]
        centers_of_mass.append(atoms[molIdxs_i].get_center_of_mass())
    return centers_of_mass

def unwrap_atoms(structure_file=None, write_traj=False):
    if write_traj:
        traj_writer = Trajectory('traj.traj','w')
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
    print("The following atoms are part of molecule {}: {}".format(molIdx, molIdxs))
    edges = list(matrix.keys())
    max_bond_len = max(natural_cutoffs(atoms))
    cell = list(atoms.get_cell())
    cell_array = atoms.get_cell()
    atoms = read(structure_file)
    atoms.set_pbc([False,False,False])
    if write_traj:
        traj_writer.write(atoms)
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
                    if write_traj:
                        atoms.set_positions(all_positions)
                        traj_writer.write(atoms)
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
                if test_weight < weight and not has_dupes and is_coplanar(centers_of_mass):
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
    if write_traj:
        traj_writer.write(atoms)

    new_atoms = Atoms()
    new_atoms.set_cell(cell_array)

    atom_mapping = {}
    counter = 0
    for i in range(n_components):
        molIdx = i
        molIdxs = [ x for x in range(len(component_list)) if component_list[x] == molIdx ]
        for idx in molIdxs:
            atom_mapping[idx] = counter 
            counter += 1
        new_atoms.extend(atoms[molIdxs])
    
    with open('atom_mapping.json', 'w') as f:
        f.write(json.dumps(atom_mapping, sort_keys=True, indent=2))

    fully_connected_atoms = new_atoms*[2, 2, 2]

    n_components, component_list, edges = find_neighbors(fully_connected_atoms)

    # Compute centers of mass for remaining molecules
    centers_of_mass = get_centers_of_mass(fully_connected_atoms, n_components, component_list)
    com_edges = {}
    for i in trange(n_components, desc="Finding 3 closest fully-connected molecules"):
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

    return pairs

    



# unwrap_atoms("tloc/rearrange_atoms/bdt.cif", write_traj=False)