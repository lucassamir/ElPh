from ase.io import read, write
from ase.io.gaussian import write_gaussian_in
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy import sparse
import numpy as np
import os
from ase.io.trajectory import Trajectory
from ase import Atoms
import json
from tqdm.auto import trange, tqdm
import warnings
from halo import Halo
from ase.calculators.gaussian import Gaussian

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
    centers_of_mass = []
    for i in range(n_components):
        molIdx_i = i
        molIdxs_i = [ x for x in range(len(component_list)) if component_list[x] == molIdx_i ]
        centers_of_mass.append(atoms[molIdxs_i].get_center_of_mass())
    return centers_of_mass

def write_structure(label, component_list, molecules, all_atoms):
    if isinstance(molecules, list):
        idxs = [ i for i in range(len(component_list)) if component_list[i] in molecules ]
    else:
        idxs = [ i for i in range(len(component_list)) if component_list[i] == molecules ]
    atoms = all_atoms[idxs]
    atoms.set_pbc([False, False, False])

    os.mkdir(label)
    outfile = open(f"{label}/{label}.com", 'w')
    write_gaussian_in(outfile, 
                        atoms,
                        nprocshared=24,
                        mem="48GB",
                        method="b3lyp",
                        basis="6-31G*",
                        scf="tight",
                        extra="nosymm punch=mo iop(3/33=1)")

def unwrap_atoms(structure_file=None):
    spinner = Halo(text="Reading structure", color='blue', spinner='dots')
    spinner.start()
    folder = os.getcwd()

    if structure_file:
        if os.path.exists(folder + '/../' + structure_file):
            structure_file = folder + '/../' + structure_file
        else:
            structure_file = folder + '/' + structure_file
    else:
        structure_file = find_structure_file(folder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = read(structure_file)
    atoms *= [3, 3, 3]
    spinner.stop()
    spinner = Halo(text="Identifying molecules", color='green', spinner='dots')
    spinner.start()
    neighbor_list = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix(neighbor_list.nl)
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    edges = list(matrix.keys())
    spinner.stop()

    # Compute total weight of each molecule and record minimum weight
    positions = atoms.get_positions()
    weights = []
    min_weight = float('inf')
    for idx in trange(n_components, desc="Finding fully-connected molecules"):
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
    for i, weight in enumerate(tqdm(weights, desc="Discarding non-connected molecules")):
        if weight <= min_weight + 1:
            keep_idx.append(i)
    keep_idxs = [ i for i in range(len(component_list)) if component_list[i] in keep_idx ]
    fully_connected_atoms = atoms[keep_idxs]

    spinner = Halo(text="Re-identifying molecules", color='green', spinner='dots')
    spinner.start()
    # Re-compute molecules so that they fall in order
    neighbor_list = NeighborList(natural_cutoffs(fully_connected_atoms), self_interaction=False, bothways=True)
    neighbor_list.update(fully_connected_atoms)
    matrix = neighbor_list.get_connectivity_matrix(neighbor_list.nl)
    n_components, component_list = sparse.csgraph.connected_components(matrix)

    # Compute centers of mass for remaining molecules
    centers_of_mass = get_centers_of_mass(fully_connected_atoms, n_components, component_list)
    spinner.stop()
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
    print(f"Minimum cycle: {min_cycle} With weight: {min_cycle_length}")

    # Keep only these atoms
    keep_idxs = [ i for i in range(len(component_list)) if component_list[i] in min_cycle ]
    new_atoms = fully_connected_atoms[keep_idxs]

    # Center in cell
    new_atoms.center()
    new_atoms.set_pbc([False, False, False])
    new_atoms.calc = Gaussian(mem="48GB",
                                method="b3lyp",
                                basis="6-31G*",
                                scf="tight",
                                
                                extra="nosymm punch=mo iop(3/33=1)")
    outfile = open("full_structure.com", 'w')
    write_gaussian_in(outfile, 
                        new_atoms,
                        nprocshared=24,
                        mem="48GB",
                        method="b3lyp",
                        basis="6-31G*",
                        scf="tight",
                        pop="full",
                        extra="nosymm punch=mo iop(3/33=1)")

    # Create structures with each pair of atoms
    """
    Directory structure will be as follows:
    >main_folder
        >0
            -0.com
            ~0.log
            ~fort.7 > 0.pun
            >Displacements
                -0
                -1
                ...
        >1
            -1.com
            ~1.log
            ~fort.7 > 1.pun
        >2
            -C.com
            ~C.log
            ~fort.7 > C.pun
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

    # A
    write_structure('A', component_list, min_cycle[0], fully_connected_atoms)
    # B
    write_structure('B', component_list, min_cycle[1], fully_connected_atoms)
    # C
    write_structure('C', component_list, min_cycle[2], fully_connected_atoms)
    # AB
    write_structure('AB', component_list, [min_cycle[0], min_cycle[1]], fully_connected_atoms)
    # BC
    write_structure('BC', component_list, [min_cycle[1], min_cycle[2]], fully_connected_atoms)
    # AC
    write_structure('AC', component_list, [min_cycle[0], min_cycle[2]], fully_connected_atoms)
