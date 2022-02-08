import numpy as np
from pathlib import Path

class Structure:
    '''This class defines the charge mobility of molecular 
       semiconductor through the Transient Localization Theory'''

    def __init__(self, lattice, params):
        '''Creates the structure object.
        
        lattice: dict
           Lattice parameters as well as the size of the supercell
        params: dict
           Hopping averages (J) and fluctuations (DeltaJ), 
           temperature and inverse of tau_in
        '''

        lat_dic = read_json('lattice.json')
        nmol = lat_dic['nmol']
        coordmol = lat_dic['coordmol']
        unitcell = lat_dic['unitcell']
        supercell = lat_dic['supercell']
        unique = lat_dic['unique']
        interactions = lat_dic['interactions']

        par_dic = read_json('params.json')
        javg = par_dic['javg']
        jdelta = par_dic['jdelta']
        nrepeat = par_dic['nrepeat']
        iseed = par_dic['iseed']
        invtau = par_dic['invtau']
        temp = par_dic['temp']
        
    def write_json(filename, data):
        """Creates json file with specified name and data.
 
        Args: 
           filename (string): Name of json file. Must include .json.
           data (dict): Dictionary with data.
        """
        Path(filename).write_text(jsonio.MyEncoder(indent=4).encode(data))
        
    def read_json(filename):
        """Reads specified json file. 

        Args:
           filename (str): Full name of json file.
        Returns:
           dict: Dictionary with data specified by json file.
        """
        dct = jsonio.decode(Path(filename).read_text())
        return dct

    def squared_length(self):
