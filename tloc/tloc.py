import numpy as np
import json
from scipy import linalg
from pathlib import Path

class Structure:
   def __init__(self, lattice_file, params_file, 
   nmuc=None, coordmol=None, unitcell=None, 
   supercell=None, unique=None, uniqinter=None,
   javg=None, jdelta=None, nrepeat=None,
   invtau=None, temp=None):

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
      self.invtau = par_dic['invtau']
      self.temp = par_dic['temp']

      # addind 0 for the case that molecules dont interact
      self.javg = np.insert(self.javg, 0, 0)
      self.jdelta = np.insert(self.jdelta, 0, 0)

   def get_interactions(self):
      # nmol number of molecules in the supercell
      nmol = self.nmuc * self.supercell[0] * self.supercell[1] \
      * self.supercell[2]
      
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
      transcoords_mk = (transvecs_tk[:, None] \
         + self.coordmol[None, :]).reshape(nmol, 3)

      # distances between all molecules THAT INTERACT and enforces PBC
      distx_mm = transcoords_mk[:, 0, None] - transcoords_mk[None, :, 0]
      disty_mm = transcoords_mk[:, 1, None] - transcoords_mk[None, :, 1]

      index = np.where(transinter_mm == 0)
      distx_mm[index] = 0
      disty_mm[index] = 0

      superlengthx = self.unitcell[0, 0] * self.supercell[0]
      superlengthy = self.unitcell[1, 1] * self.supercell[1]

      distx_mm[distx_mm > superlengthx / 2] -= superlengthx
      distx_mm[distx_mm < -superlengthx / 2] += superlengthx
      disty_mm[disty_mm > superlengthy / 2] -= superlengthy
      disty_mm[disty_mm < -superlengthy / 2] += superlengthy

      return nmol, transinter_mm, distx_mm, disty_mm

   def get_hamiltonian(self, nmol, transinter_mm):
      rnd1_mm = np.random.rand(nmol, nmol)
      rnd1_mm = (rnd1_mm + rnd1_mm.T) / 2 # enforce hermitian
      rnd2_mm = np.random.rand(nmol, nmol)
      rnd2_mm = (rnd2_mm + rnd2_mm.T) / 2 # enforce hermitian

      log_mm = -2 * np.log(1 - rnd1_mm)
      cos_mm = np.sqrt(log_mm) * np.cos(2 * np.pi * rnd2_mm)
      # sin = np.sqrt(log) * np.sin(2 * np.pi * rnd2)

      hamiltonian_mm = self.javg[transinter_mm] + self.jdelta[transinter_mm] * cos_mm
      # y = javg + (jdelta * sin)     
    
      return hamiltonian_mm

   def get_energies(self, nmol, transinter_mm):
      hamiltonian_mm = self.get_hamiltonian(nmol, transinter_mm)

      energies_m, vectors_mm = linalg.eigh(hamiltonian_mm)
      
      return energies_m.real, vectors_mm, hamiltonian_mm

   def get_squared_length(self):
      nmol, transinter_mm, distx_mm, disty_mm = self.get_interactions()

      energies_m, vectors_mm, hamiltonian_mm = self.get_energies(nmol, transinter_mm)

      energies_m = np.array([-0.20138311553842075, 
      -0.12440805722822261,
      -0.10049837807332004,
      -6.4391947830542390E-002,
      -3.3528173872373460E-002,
      7.3905024331541683E-002,
      0.11264972606239899,
      0.33765492214893883])

      vectors_mm = np.array([[0.30419710108443671,      -0.34726236245266096,       0.58051715228326739,      -0.12669215720015056   ,   -0.28963626132521336  ,      8.5308983596902949E-002, -0.44663762119164674 ,      0.37837806134077467],
      [-0.13141439300172358 ,      -8.2668840398545870E-002 , 0.22232750768723919    ,   0.51363143871214045  ,     0.62972998330982166     , -0.41648774284801132     , -0.14167960413323802     ,  0.26935913309530762],
      [0.58229203523993889 ,      0.36328132135808738 ,       8.1617898886527054E-002 ,-0.20553268126367571      , 0.11808585507072468    ,  -0.19664507162755132  ,     0.51045772575221238    ,   0.40850564208962270],
      [-0.33424858464572688     ,  0.53534795782531153      , 0.21179120712954377  ,     0.31142330212068936    ,  -0.11117737152079106      , 0.55586149027625842   ,     3.5184580281843417E-002 , 0.37048630230202667],
      [0.30756577249783545 ,      -9.5053032830171875E-003 ,-0.72391387063081114     ,  0.16553688539272798       , 1.7747033703785532E-002 , 0.14647230146294482   ,   -0.42276189517074014  ,     0.39161521927857873],
      [-0.54437968067735965   ,    0.13973521703728831     , -0.13264547787016273  ,    -0.47427661516393521      ,-0.17690279431565051,      -0.45870952367326223 ,     -0.12704042346417735     ,  0.42865301463444888],
      [-9.9533054473112823E-002 ,-0.42570873440971135    ,  -0.12374122295572730   ,    0.46180033777147417    ,  -0.51175528325828401  ,    -0.15876968068147182  ,     0.48197078972245644     ,  0.24677252419578066],
      [-0.19635549465943125 ,     -0.50314983040031747    ,   -7.1381462893388550E-002, -0.33956349364522065     ,  0.44692648316627426 ,      0.46317672987682074    ,   0.30244660568633364   ,    0.28659409393407004]
      ])

      test = energies_m[:, None] - energies_m[None, :]

      operatorx_mm = (energies_m[:, None] - energies_m[None, :]) * \
         np.dot(vectors_mm.T, np.dot(distx_mm, vectors_mm))
      operatory_mm = (energies_m[:, None] - energies_m[None, :]) * \
         np.dot(vectors_mm.T, np.dot(disty_mm, vectors_mm))
      
      partfunc_m = np.exp(energies_m / self.temp)
      partfunc = sum(partfunc_m)

      sqlx = sum(sum(partfunc_m * (operatorx_mm**2 * 2 / (self.invtau**2 + \
         (energies_m[:, None] - energies_m[None, :])**2))))
      sqly = sum(sum(partfunc_m * (operatory_mm**2 * 2 / (self.invtau**2 + \
         (energies_m[:, None] - energies_m[None, :])**2))))

      #print(partfunc, sqlx)
      sqlx /= partfunc
      sqly /= partfunc

      return sqlx, sqly

   def get_disorder_avg_sql(self):
      dsqlx, dsqly = 0, 0

      print('Calculating average of squared transient localization')
      for i in range(1, self.nrepeat + 1):
         sqlx, sqly = self.get_squared_length()
         #tsqlx = self.get_therm_avg(sqlx)
         #tsqly = self.get_therm_avg(sqly)
         #partfunc = self.get_therm_avg()

         #moving average
         dsqlx -= dsqlx / i
         dsqlx += sqlx / i

         dsqly -= dsqly / i
         dsqly += sqly / i

         print(i, dsqlx, dsqly)

      return dsqlx, dsqly
   
   def get_mobility(self):
      dsqlx, dsqly = self.get_disorder_avg_sql()

      mobx = (1 / self.temp) * self.invtau * dsqlx / 2
      moby = (1 / self.temp) * self.invtau * dsqly / 2

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
   st = Structure(args.lattice_file, args.params_file)

   if args.mobility:
      mobx, moby = st.get_mobility()
      print('Calculating charge mobility')
      print('mu_x = ', mobx)
      print('mu_y = ', moby)

if __name__  == '__main__':
   main()