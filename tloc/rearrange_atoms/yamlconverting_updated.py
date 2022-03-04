# -*- coding: utf-8 -*-

import yaml
from yaml import CLoader as Loader
from ase.io import read, write
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs
from ase import Atoms
# from ase.utils import natural_cutoffs
from datetime import datetime

t1 = datetime.now()

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


def bymolecule(filename_yaml, filename_POSCAR_arranged, new_filename,filename_POSCAR_original, **kwargs):
    with open(filename_yaml, 'r') as stream:
        parsed = yaml.load(stream, Loader=Loader)
    f = open(new_filename, "w")
    f.write("mesh: " + str(parsed['mesh']) + '\n') #prints mesh line in new file
    f.write("nqpoint:   " + str(parsed['nqpoint']) + '\n') #prints nqpoint line in new file
    f.write("reciprocal_lattice:" + '\n') #prints mesh line in new file
     #prints reciprocal lattice in new file
    f.write('- [' + '{:13.8f},{:13.8f},{:13.8f}'.format(parsed['reciprocal_lattice'][0][0],
                                                       parsed['reciprocal_lattice'][0][1],
                                                       parsed['reciprocal_lattice'][0][2]) + ' ] # a*' + '\n')
    f.write('- [' + '{:13.8f},{:13.8f},{:13.8f}'.format(parsed['reciprocal_lattice'][1][0],
                                                       parsed['reciprocal_lattice'][1][1],
                                                       parsed['reciprocal_lattice'][1][2]) + ' ] # b*' + '\n')
    f.write('- [' + '{:13.8f},{:13.8f},{:13.8f}'.format(parsed['reciprocal_lattice'][2][0],
                                                       parsed['reciprocal_lattice'][2][1],
                                                       parsed['reciprocal_lattice'][2][2]) + ' ] # c*' + '\n')
    
    f.write("natom: " + str(parsed['natom']) + '\n') #prints mesh line in new file
    f.write("lattice: " + '\n') #prints lattice in new file
    
    f.write('- [' + '{:22.15f},{:22.15f},{:22.15f}'.format(parsed['lattice'][0][0],
                                                       parsed['lattice'][0][1],
                                                       parsed['lattice'][0][2]) + ' ] # a' + '\n')
    f.write('- [' + '{:22.15f},{:22.15f},{:22.15f}'.format(parsed['lattice'][1][0],
                                                       parsed['lattice'][1][1],
                                                       parsed['lattice'][1][2]) + ' ] # b' + '\n')
    f.write('- [' + '{:22.15f},{:22.15f},{:22.15f}'.format(parsed['lattice'][2][0],
                                                       parsed['lattice'][2][1],
                                                       parsed['lattice'][2][2]) + ' ] # c' + '\n')
    
    f.write("points: " + '\n') #rearranges and prints atoms positions by molecule in new file
    atoms = read(filename_POSCAR_arranged) #reads the POSCAR file
    for i in range(len(atoms)):
        f.write('- symbol: ' + atoms.get_chemical_symbols()[i] +'   # ' + str(i+1) + '\n')
        f.write('  coordinates: [' + '{:19.15f}, {:19.15f}, {:19.15f} ] \n'.format(atoms.get_scaled_positions(wrap=False)[i][0],
                                                                            atoms.get_scaled_positions(wrap=False)[i][1],
                                                                            atoms.get_scaled_positions(wrap=False)[i][2]))
        f.write('  mass: ' + str(atoms.get_masses()[i]) + '\n')

    atoms_original = read(filename_POSCAR_original)
    molecules = tag_linkers(atoms_original)
    molecules_list = []
    for i in range(len(molecules)):
        for j in range(len(molecules[i])):
            molecules_list.append(molecules[i][j])
    span = range(len(parsed['phonon'][0]['band'])) #range of the number of modes
    for j in range(len(parsed['phonon'])):
        f.write('\n' + "phonon: " + '\n')
        f.write("- q-position: " + str(parsed['phonon'][j]['q-position']) + '\n')
        f.write("  distance_from_gamma: " + str(parsed['phonon'][j]['distance_from_gamma']) + '\n')
        f.write("  weight: " + str(parsed['phonon'][j]['weight']) + '\n')
        f.write("  band: " + '\n')
        for i in span: #reorganizing atoms by molecule in each mode
            count = 1
            f.write("  - # " + str(i+1) + '\n')
            f.write("    frequency: " + str(parsed['phonon'][j]['band'][i]['frequency']) + '\n')
            f.write("    eigenvector: " + '\n')
            for k in molecules_list:
                f.write("    - # atom " + str(count) + '\n')
                for m in range(len(parsed['phonon'][j]['band'][0]['eigenvector'][0])):
                    f.write('      - ['+ str('%18.14f' % parsed['phonon'][j]['band'][i]['eigenvector'][k][m][0]))
                    f.write(', '+ str('%18.14f' % parsed['phonon'][j]['band'][i]['eigenvector'][k][m][1]) + ' ] \n')
                count = count + 1

'''The 3 files needed to run'''
filename_POSCAR_original = 'POSCAR' #original unmanipulated POSCAR
filename_yaml = 'mesh.yaml' #original yaml.file
filename_POSCAR_arranged = 'POSCAR_new' #The rearranged POSCAR file
new_filename = 'rubrene_by_molecule_mesh.yaml' #New filename for new yaml file
bymolecule(filename_yaml, filename_POSCAR_arranged, new_filename, filename_POSCAR_original)

t2 = datetime.now()


