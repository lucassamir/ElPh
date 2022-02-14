import numpy as np
from pathlib import Path

class Structure:
   def __init__(self, lattice_file, params_file, 
   nmuc=None, coordmol=None, unitcell=None, 
   supercell=None, unique=None, uniqinter=None,
   javg=None, jdelta=None, nrepeat=None,
   iseed=None, invtau=None, temp=None):

      lat_dic = read_json(lattice_file)
      self.nmuc = lat_dic['nmuc']
      self.coordmol = np.array(lat_dic['coordmol'])
      self.unitcell = np.array(lat_dic['unitcell'])
      self.supercell = lat_dic['supercell']
      self.unique = lat_dic['unique']
      self.uniqinter = np.array(lat_dic['uniqinter'])

      par_dic = read_json(params_file)
      self.javg = par_dic['javg']
      self.jdelta = par_dic['jdelta']
      self.nrepeat = par_dic['nrepeat']
      self.iseed = par_dic['iseed']
      self.invtau = par_dic['invtau']
      self.temp = par_dic['temp']
        
   def write_json(filename, data):
      Path(filename).write_text(jsonio.MyEncoder(indent=4).encode(data))
        
   def read_json(filename):
      dct = jsonio.decode(Path(filename).read_text())
      return dct

   def get_distances(self):
      nmol = self.nmuc * self.supercell[0] * self.supercell[1] \
      * self.supercell[2]
      
      superlengthx = self.unitcell[0, 0] * self.supercell[0]
      superlengthy = self.unitcell[1, 1] * self.supercell[1]

      # t number of translations and k directions
      translations_tk = np.mgrid[0:self.supercell[0]:1, 
                                 0:self.supercell[1]:1, 
                                 0:self.supercell[2]:1].reshape(3,-1).T
      transvecs_tk = np.matmul(translations_tk, self.unitcell)

      # n number of connections
      nconnect = self.unique * len(transvecs_tk)

      # translated coordinates for m molecules
      transcoords_mtk = self.coordmol[:, None] + transvecs_tk[None, :]

      # translated interactions
      transinter = np.empty([nmol, 3])
      transinter[:, 0] = np.tile(self.uniqinter[:, 0], nconnect) \
         + np.repeat(np.arange(0, nmol, self.nmuc), self.unique)
      transinter[:, 1] = np.tile(self.uniqinter[:, 1], nconnect) \
         + np.repeat(np.arange(0, nmol, self.nmuc), self.unique) \
            + self.uniqinter[:, 2:5][None, :] * self.supercell
      

   def squared_length(self):
      nmol, nconnect, transinter, distx, disty = get_distances()
