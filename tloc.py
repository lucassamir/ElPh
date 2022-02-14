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

   def get_interactions(self):
      #nmol number of molecules in the supercell
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

      # connections between molecules
      connec = (translations_tk[:, None] \
         + self.uniqinter[:, 2:5][None, :]).reshape(nconnect, 3)

      # enforce PBC
      connec[:, 0][connec[:, 0] == self.supercell[0]] = 0
      connec[:, 1][connec[:, 1] == self.supercell[1]] = 0
      connec[:, 2][connec[:, 2] == self.supercell[2]] = 0
      
      # translated interactions 
      transinter = np.empty([nconnect, 3])
      transinter[:, 0] = (np.arange(0, nmol, self.nmuc)[:, None] \
         + self.uniqinter[:, 0][None, :]).reshape(nconnect)
      transinter[:, 1] = np.tile(self.uniqinter[:, 1], t) \
         + self.nmuc * np.matmul(connec,
         [self.supercell[1] * self.supercell[2], self.supercell[2], 1])
      transinter[:, 2] = np.tile(self.uniqinter[:, 2], t)

      superlengthx = self.unitcell[0, 0] * self.supercell[0]
      superlengthy = self.unitcell[1, 1] * self.supercell[1]

      # translated coordinates for m molecules
      transcoords_mtk = self.coordmol[:, None] + transvecs_tk[None, :]

      # distance between molecules
      distx = transcoords_mtk

   def squared_length(self):
      nmol, nconnect, transinter, distx, disty = get_interactions()
