# -*- coding: utf-8 -*-

from ase.io import read, write
from ase.visualize import view
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase import build

POSCAR = 'POSCAR' #Enter the filename of the POSCAR file

def tag_linkers(atoms):
    '''
    Takes atoms object of disconnected subsets (e.g. linkers in MOFs) and assigns a unique tag to each
    subset. Returns list of nested lists with indices corresponding to each tag
    '''
    nl_linkers = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl_linkers.update(atoms)
    atoms.set_tags(0) #initialize all tags to 0
    tag_index = 1
    linkers_indices = []

    # Grabs atom from subset, tags all connected atoms
    # with tag_index until all atoms in subset are tagged. Repeats until no atoms are untagged
    while 0 in atoms.get_tags():
        possible_atoms = [a.index for a in atoms if a.tag == 0] # all current untagged atoms
        neighbor_indices = [possible_atoms[0]] # grabs first untagged atom
        for i in neighbor_indices: # finds all connected atoms, tags, and adds to list of iterables
            new_atoms = [k for k in list(nl_linkers.get_neighbors(i)[0]) if atoms[k].tag == 0]
            for j in new_atoms:
                atoms[j].tag = tag_index
            neighbor_indices.extend(new_atoms)

        linkers_indices.append([a.index for a in atoms if a.tag == tag_index])
        tag_index = tag_index + 1
    return linkers_indices

''' This section of code creates the whole 
molecule(s) in a large cell to visualize easily '''

atoms = read(POSCAR)
atoms3 = atoms.repeat((4,4,3)) 
atoms.set_cell(atoms3.get_cell()) 
molecules = tag_linkers(atoms3)

'''This section reindexes the atoms for later'''
count = 0
for i in range(len(atoms3)):
    atoms3[i].tag = atoms[count].index
    count += 1
    if count == len(atoms):
        count = 0

''' We have 4 molecules, so we want 4 molecules close to each other. Here I swept through and found 116, 109, 108, and 107 are connected'''
n = 55
atomsS = Atoms()
atomsS.set_cell(atoms3.get_cell())
for j in [116, 109, 108, 107]: #I use a for loop to sweep through about 10 different n then go to the next one etc etc
    for i in range(len(molecules[1])):
        atomsS.append(atoms3[molecules[j][i]])
view(atomsS) #View molecules to make sure its right 

'''Collects the correct atoms from the supercell from the check above'''
tags = atomsS.get_tags()
atomsF = Atoms()
for i in range(len(tags)):
    a = True
    count = 0
    while count != len(tags):
        if tags[count] == i:
            atomsF.append(atomsS[count])
        count += 1

atoms = atomsF

atoms1 = read(POSCAR)
atoms.set_cell(atoms1.get_cell())
atoms.set_pbc([False,False,False])
'''You may need to change which atom to translate from. Most times atom[0] is perfectly fine '''
trans = atoms1.get_positions()[66]-atoms.get_positions()[66] 
atoms.translate(trans)

'''This section makes a new POSCAR with the atoms by molecule.'''

molecules = tag_linkers(atoms)
new_atoms = Atoms()
for i in range(len(molecules)):
    for j in range(len(molecules[0])):
        new_atoms.append(atoms[molecules[i][j]])
new_atoms.set_cell(atoms.get_cell())
write('POSCAR_new', new_atoms, format='vasp', direct=True) 
write('POSCAR_new.xyz', new_atoms) #Troisis student wants files in xyz format for gaussian calculations