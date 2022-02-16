import numpy as np
import json
from scipy.sparse import linalg
from pathlib import Path

class Structure:
   def __init__(self, lattice_file, params_file, 
   nmuc=None, coordmol=None, unitcell=None, 
   supercell=None, unique=None, uniqinter=None,
   javg=None, jdelta=None, nrepeat=None,
   iseed=None, invtau=None, temp=None):

      with open(lattice_file + '.json', 'r') as json_file:
         lat_dic = json.load(json_file)
      self.nmuc = lat_dic['nmuc']
      self.coordmol = np.array(lat_dic['coordmol'])
      self.unitcell = np.array(lat_dic['unitcell'])
      self.supercell = lat_dic['supercell']
      self.unique = lat_dic['unique']
      self.uniqinter = np.array(lat_dic['uniqinter'])

      with open(params_file + '.json', 'r') as json_file:
         par_dic = json.load(json_file)
      self.javg = np.array(par_dic['javg'])
      self.jdelta = np.array(par_dic['jdelta'])
      self.nrepeat = par_dic['nrepeat']
      self.iseed = par_dic['iseed']
      self.invtau = par_dic['invtau']
      self.temp = par_dic['temp']

   def get_interactions(self):
      # nmol number of molecules in the supercell
      nmol = self.nmuc * self.supercell[0] * self.supercell[1] \
      * self.supercell[2]
      
      # t number of translations and k directions
      translations_tk = np.mgrid[0:self.supercell[0]:1, 
                                 0:self.supercell[1]:1, 
                                 0:self.supercell[2]:1].reshape(3,-1).T
      transvecs_tk = np.matmul(translations_tk, self.unitcell)
      t = len(transvecs_tk)

      # nconnect number of connections
      nconnect = self.unique * t

      # mapping
      mapcell_k = [self.supercell[1] * self.supercell[2], 
                   self.supercell[2], 
                   1]
      map2same_t = self.nmuc * np.matmul(translations_tk, mapcell_k)

      # u unique connections to other molecules and enforced PBC
      connec_tuk = (translations_tk[:, None] \
         + self.uniqinter[:, 2:5][None, :])
      connec_tuk[:, :, 0][connec_tuk[:, :, 0] == self.supercell[0]] = 0
      connec_tuk[:, :, 1][connec_tuk[:, :, 1] == self.supercell[1]] = 0
      connec_tuk[:, :, 2][connec_tuk[:, :, 2] == self.supercell[2]] = 0
      map2others_tu = self.nmuc * np.matmul(connec_tuk, mapcell_k)
      
      # translated interactions 
      transinter = np.empty([nconnect, 3], dtype='int')
      transinter[:, 0] = (map2same_t[:, None] \
         + self.uniqinter[:, 0][None, :]).reshape(nconnect)
      transinter[:, 1] = (map2others_tu[:, None] \
         + self.uniqinter[:, 1][None, :]).reshape(nconnect)
      transinter[:, 2] = np.tile(self.uniqinter[:, 5], t)

      # translated coordinates for m molecules
      transcoords_mk = (transvecs_tk[:, None] \
         + self.coordmol[None, :]).reshape(nmol, 3)

      # n distances between molecules and enforces PBC
      distx_n = transcoords_mk[transinter[:, 0] - 1, 0] \
         - transcoords_mk[transinter[:, 1] - 1, 0]
      disty_n = transcoords_mk[transinter[:, 0] - 1, 1] \
         - transcoords_mk[transinter[:, 1] - 1, 1]

      superlengthx = self.unitcell[0, 0] * self.supercell[0]
      superlengthy = self.unitcell[1, 1] * self.supercell[1]

      distx_n[distx_n > superlengthx / 2] -= superlengthx
      distx_n[distx_n < -superlengthx / 2] += superlengthx
      disty_n[disty_n > superlengthy / 2] -= superlengthy
      disty_n[disty_n < -superlengthy / 2] += superlengthy

      return nmol, nconnect, transinter, distx_n, disty_n

   def get_hamiltonian(self, nmol, transinter):
      rnd1_n = np.random.rand(len(transinter))
      rnd2_n = np.random.rand(len(transinter))
    
      log_n = -2 * np.log(1 - rnd1_n)
      cos_n = np.sqrt(log_n) * np.cos(2 * np.pi * rnd2_n)
      # sin = np.sqrt(log) * np.sin(2 * np.pi * rnd2)
    
      # populate sparse hamiltonian with just interactions 
      # between transinter[0] and transinter[1] molecules
      hamiltonian_mm = np.zeros([nmol, nmol])
      hamiltonian_mm[transinter[:, 0] - 1, transinter[:, 1] - 1] = \
         self.javg[transinter[:, 2] - 1] \
         + (self.jdelta[transinter[:, 2] - 1] * cos_n)

      # assert that Hamiltonian is Hermitian
      hamiltonian_mm[transinter[:, 1] - 1, transinter[:, 0] - 1] = \
         self.javg[transinter[:, 2] - 1] \
         + (self.jdelta[transinter[:, 2] - 1] * cos_n) # Corina: I use self.

      # y = javg + (jdelta * sin)     
    
      return hamiltonian_mm

   def get_energies(self, nmol, transinter):
      hamiltonian_mm = self.get_hamiltonian(nmol, transinter)

      energies, vectors = linalg.eigsh(hamiltonian_mm)

      return energies, vectors

   def get_squared_length(self):
      nmol, nconnect, transinter, distx_n, disty_n = self.get_interactions()

      energies, vectors = self.get_energies(nmol, transinter)

      operatorx = distx_n * vectors * vectors * \
         (energies - energies)

      operatory = disty_n * vectors * vectors * \
         (energies - energies)

      sqlx = operatorx**2 * 2 / (self.invtau**2 + (energies - energies)**2)
      sqly = operatory**2 * 2 / (self.invtau**2 + (energies - energies)**2)

      return sqlx, sqly

   def get_disorder_avg_sql(self):
      dsqlx, dsqly = 0, 0

      print('Calculating average of squared transient localization')
      for i in range(self.nrepeat):
         sqlx, sqly = self.get_squared_length()
         tsqlx = self.get_therm_avg(sqlx)
         tsqly = self.get_therm_avg(sqly)
         partfunc = self.get_therm_avg()

         dsqlx = (1 - 1 / i) * dsqlx + (tsqlx / partfunc) / i
         dsqly = (1 - 1 / i) * dsqly + (tsqly / partfunc) / i

         print(i, dsqlx, dsqly)

      return dsqlx, dsqly
   
   def get_mobility(self):
      dsqlx, dsqly = self.get_disorder_avg_sql()

      mobx = self.temp * self.invtau * dsqlx / 2
      moby = self.temp * self.invtau * dsqly / 2

      return mobx, moby

def write_lattice_file():
   lattice = {'nmuc':2,
              'coordmol':[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
              'unitcell':[[1.0, 0.0, 0.0], [0.0, 1.7321, 0.0], [0.0, 0.0, 1000.0]],
              'supercell':[16, 16, 1],
              'unique':6,
              'uniqinter':[[1, 1, 1, 0, 0, 1], 
              [2, 2, 1, 0, 0, 1], 
              [1, 2, 0, 0, 0, 3], 
              [2, 1, 1, 0, 0, 2], 
              [2, 1, 0, 1, 0, 2], 
              [2, 1, 1, 1, 0, 3]]
   }
   with open('lattice.json', 'w', encoding='utf-8') as f:
      json.dump(lattice, f, ensure_ascii=False, indent=2)

def write_params_file():
   params = {'javg':[-0.98296, 0.12994, 0.12994],
             'jdelta':[0.49148, 0.06497, 0.06497],
             'nrepeat':50,
             'iseed':3987187,
             'invtau':0.05,
             'temp':0.25
   }
   with open('params.json', 'w', encoding='utf-8') as f:
      json.dump(params, f, ensure_ascii=False, indent=4)

def main(args=None):
   import argparse

   description = "Transient Localization Theory command line interface"

   example_text = """examples:

   Calculate charge mobility with:
      tloc --mobility
   """

   formatter = argparse.RawDescriptionHelpFormatter
   parser = argparse.ArgumentParser(description=description,
                                    epilog=example_text, 
                                    formatter_class=formatter)

   help = """
   All calculations require a lattice JSON 
   file with the following properties:

   lattice.json:

   """
   parser.add_argument('--lattice_file', nargs='*', help=help,
                        default='lattice', type=str)

   help = """
   All calculations require a params json 
   file with the following properties:

   params.json:
   
   """
   parser.add_argument('--params_file', nargs='*', help=help,
                        default='params', type=str)

   help = ("write example of lattice and params files")
   parser.add_argument('--write_examples', action='store_true' , help=help)

   help = ("Calculate charge mobility")
   parser.add_argument('--mobility', action='store_true' , help=help)

   args = parser.parse_args(args)

   if args.write_examples:
      write_lattice_file()
      write_params_file()
      return

   if not Path(args.lattice_file + '.json').is_file():
      msg = 'Lattice file could not be found'
      raise FileNotFoundError(msg)

   if not Path(args.params_file + '.json').is_file():
      msg = 'Params file could not be found'
      raise FileNotFoundError(msg)

   print('Initializing molecular structure')
   st = Structure(args.lattice_file, args.params_file)

   if args.mobility:
      print('Calculating charge mobility')
      st.get_mobility()

if __name__  == '__main__':
   main()