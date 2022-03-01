import numpy as np
import json
from scipy import linalg
from pathlib import Path

class Structure:
   def __init__(self, nmuc=None, coordmol=None, unitcell=None, 
   supercell=None, unique=None, uniqinter=None,
   javg=None, jdelta=None, nrepeat=None,
   iseed=None, invtau=None, temp=None, 
   lattice_file=None, params_file=None, results={}):

      if lattice_file:
         with open(lattice_file + '.json', 'r') as json_file:
            lat_dic = json.load(json_file)
         self.nmuc = lat_dic['nmuc']
         self.coordmol = np.array(lat_dic['coordmol'])
         self.unitcell = np.array(lat_dic['unitcell'])
         self.supercell = lat_dic['supercell']
         self.unique = lat_dic['unique']
         self.uniqinter = np.array(lat_dic['uniqinter'])
      else:
         self.nmuc = nmuc
         self.coordmol = np.array(coordmol)
         self.unitcell = np.array(unitcell)
         self.supercell = supercell
         self.unique = unique
         self.uniqinter = np.array(uniqinter)

      if params_file:
         with open(params_file + '.json', 'r') as json_file:
            par_dic = json.load(json_file)
         self.javg = np.array(par_dic['javg'])
         self.jdelta = np.array(par_dic['jdelta'])
         self.nrepeat = par_dic['nrepeat']
         self.iseed = par_dic['iseed']
         self.invtau = par_dic['invtau']
         self.temp = par_dic['temp']
      else:
         self.javg = np.array(javg )
         self.jdelta = np.array(jdelta)
         self.nrepeat = nrepeat
         self.iseed = iseed
         self.invtau = invtau
         self.temp = temp

      self.results = results

      # addind 0 for the case that molecules dont interact
      self.javg = np.insert(self.javg, 0, 0)
      self.jdelta = np.insert(self.jdelta, 0, 0)

      # making random numbers predictable
      np.random.seed(self.iseed)

   def get_interactions(self):
      # nmol number of molecules in the supercell
      nmol = self.nmuc * self.supercell[0] * self.supercell[1] * \
         self.supercell[2]
      
      # t number of translations and k directions
      translations_tk = np.mgrid[0:self.supercell[0]:1, 
                                 0:self.supercell[1]:1, 
                                 0:self.supercell[2]:1].reshape(3,-1).T
      transvecs_tk = np.dot(translations_tk, self.unitcell)
      t = len(transvecs_tk)

      # nconnect number of connections
      nconnect = self.unique * t

      # mapping
      mapcell_k = [self.supercell[1] * self.supercell[2], 
                   self.supercell[2], 
                   1]
      map2same_t = self.nmuc * np.dot(translations_tk, mapcell_k)

      # u unique connections to other molecules and enforced PBC
      connec_tuk = (translations_tk[:, None] \
         + self.uniqinter[:, 2:5][None, :])
      connec_tuk[:, :, 0][connec_tuk[:, :, 0] == self.supercell[0]] = 0
      connec_tuk[:, :, 1][connec_tuk[:, :, 1] == self.supercell[1]] = 0
      connec_tuk[:, :, 2][connec_tuk[:, :, 2] == self.supercell[2]] = 0
      map2others_tu = self.nmuc * np.dot(connec_tuk, mapcell_k)
      
      # translated interactions between a pair of molecules
      transinter_mm = np.zeros([nmol, nmol], dtype='int')
      firstmol = (map2same_t[:, None] \
         + self.uniqinter[:, 0][None, :]).reshape(nconnect)
      secondmol = (map2others_tu[:, None] \
         + self.uniqinter[:, 1][None, :]).reshape(nconnect)
      typeinter = np.tile(self.uniqinter[:, 5], t)
      transinter_mm[firstmol - 1, secondmol - 1] = typeinter

      # enforce hermitian
      transinter_mm[secondmol - 1, firstmol - 1] = typeinter

      # translated coordinates for m molecules
      transcoords_mk = (transvecs_tk[:, None] + \
         self.coordmol[None, :]).reshape(nmol, 3)

      # distances between all molecules THAT INTERACT with enforced PBC
      distx_mm = np.zeros([nmol, nmol])
      disty_mm = np.zeros([nmol, nmol])
      for i, j in zip(firstmol, secondmol):
         distx_mm[i-1, j-1] = transcoords_mk[i-1, 0] - transcoords_mk[j-1, 0]
         disty_mm[i-1, j-1] = transcoords_mk[i-1, 1] - transcoords_mk[j-1, 1]

      superlengthx = self.unitcell[0, 0] * self.supercell[0]
      superlengthy = self.unitcell[1, 1] * self.supercell[1]

      distx_mm[distx_mm > superlengthx / 2] -= superlengthx
      distx_mm[distx_mm < -superlengthx / 2] += superlengthx
      disty_mm[disty_mm > superlengthy / 2] -= superlengthy
      disty_mm[disty_mm < -superlengthy / 2] += superlengthy

      return nmol, transinter_mm, distx_mm, disty_mm

   def get_hamiltonian(self, nmol, transinter_mm):
      rnd1_mm = np.random.rand(nmol, nmol)
      rnd1_mm = np.tril(rnd1_mm) + np.tril(rnd1_mm, -1).T
      rnd2_mm = np.random.rand(nmol, nmol)
      rnd2_mm = np.tril(rnd2_mm) + np.tril(rnd2_mm, -1).T

      log_mm = -2 * np.log(1 - rnd1_mm)
      cos_mm = np.sqrt(log_mm) * np.cos(2 * np.pi * rnd2_mm)

      hamiltonian_mm = self.javg[transinter_mm] + self.jdelta[transinter_mm] * cos_mm
    
      return hamiltonian_mm

   def get_energies(self, nmol, transinter_mm):
      hamiltonian_mm = self.get_hamiltonian(nmol, transinter_mm)

      energies_m, vectors_mm = linalg.eigh(hamiltonian_mm)
      
      return energies_m.real, vectors_mm, hamiltonian_mm

   def get_squared_length(self):
      nmol, transinter_mm, distx_mm, disty_mm = self.get_interactions()

      energies_m, vectors_mm, hamiltonian_mm = self.get_energies(nmol, transinter_mm)

      operatorx_mm = np.matmul(vectors_mm.T, np.matmul(distx_mm * hamiltonian_mm, vectors_mm))
      operatorx_mm -= np.matmul(vectors_mm.T, np.matmul(distx_mm * hamiltonian_mm, vectors_mm)).T

      operatory_mm = np.matmul(vectors_mm.T, np.matmul(disty_mm * hamiltonian_mm, vectors_mm))
      operatory_mm -= np.matmul(vectors_mm.T, np.matmul(disty_mm * hamiltonian_mm, vectors_mm)).T

      partfunc_m = np.exp(energies_m / self.temp)
      partfunc = sum(partfunc_m)

      sqlx = sum(sum(partfunc_m * (operatorx_mm**2 * 2 / (self.invtau**2 + \
         (energies_m[:, None] - energies_m[None, :])**2))))
      sqly = sum(sum(partfunc_m * (operatory_mm**2 * 2 / (self.invtau**2 + \
         (energies_m[:, None] - energies_m[None, :])**2))))

      sqlx /= partfunc
      sqly /= partfunc

      return sqlx, sqly

   def get_disorder_avg_sql(self):
      dsqlx, dsqly = 0, 0
      self.results['squared_length_x'] = []
      self.results['squared_length_y'] = []
      self.results['squared_length'] = []
      self.results['avg_sqlx'] = []
      self.results['avg_sqly'] = []
      self.results['avg_sql'] = []

      print('Calculating average of squared transient localization')
      for i in range(1, self.nrepeat + 1):
         sqlx, sqly = self.get_squared_length()
         self.results['squared_length_x'].append(sqlx)
         self.results['squared_length_y'].append(sqly)
         self.results['squared_length'].append((sqlx + sqly) / 2)

         #moving average
         dsqlx -= dsqlx / i
         dsqlx += sqlx / i

         dsqly -= dsqly / i
         dsqly += sqly / i

         self.results['avg_sqlx'].append(dsqlx)
         self.results['avg_sqly'].append(dsqly)
         self.results['avg_sql'].append((dsqlx + dsqly) / 2)

         print(i, dsqlx, dsqly)

      return dsqlx, dsqly
   
   def get_mobility(self):
      dsqlx, dsqly = self.get_disorder_avg_sql()

      # unit converter unit = ang**2 * e / hbar
      unit = 0.15192674605831966

      # in units of cm**2 / (V s)
      mobx = (1 / self.temp) * self.invtau * dsqlx / 2 * unit
      moby = (1 / self.temp) * self.invtau * dsqly / 2 * unit

      self.results['mobx'] = mobx
      self.results['moby'] = moby

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
      json.dump(lattice, f, ensure_ascii=False, indent=4)

def write_params_file():
   params = {'javg':[-0.98296, 0.12994, 0.12994],
             'jdelta':[0.49148, 0.06497, 0.06497],
             'nrepeat':50,
             "iseed": 3987187,
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
   parser.add_argument('--write_files', action='store_true' , help=help)

   help = ("Calculate charge mobility")
   parser.add_argument('--mobility', action='store_true' , help=help)

   args = parser.parse_args(args)

   if args.write_files:
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
   st = Structure(lattice_file=args.lattice_file, 
      params_file=args.params_file)

   if args.mobility:
      mobx, moby = st.get_mobility()
      print('Calculating charge mobility')
      print('mu_x = ', mobx)
      print('mu_y = ', moby)
      with open('results.json', 'w', encoding='utf-8') as f:
         json.dump(st.results, f)

if __name__  == '__main__':
   main()