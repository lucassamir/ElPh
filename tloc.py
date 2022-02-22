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
         + self.uniqinter[:, 2:5][None, :]).reshape(nconnect, 3)
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

      # distance between molecules and enforces PBC
      distx = transcoords_mk[transinter[:, 0] - 1, 0] \
         - transcoords_mk[transinter[:, 1] - 1, 0]
      disty = transcoords_mk[transinter[:, 0] - 1, 1] \
         - transcoords_mk[transinter[:, 1] - 1, 1]

      superlengthx = self.unitcell[0, 0] * self.supercell[0]
      superlengthy = self.unitcell[1, 1] * self.supercell[1]

      distx[distx > superlengthx / 2] -= superlengthx
      distx[distx < -superlengthx / 2] += superlengthx
      disty[disty > superlengthy / 2] -= superlengthy
      disty[disty < -superlengthy / 2] += superlengthy
      
   def squared_length(self):
      nmol, nconnect, transinter, distx, disty = get_interactions()

   def get_jrnd(javg, jdelta):   # Assuming it is not a function of (xmed, xvar, x, y)
    
    rnd1 = np.random.random()    #discuss whether we need seed
    rnd2 = np.random.random()    #maybe not use random seed, but keep track of values used for possible reproducibility
    
    log = -2 * np.log(1 - rnd1)
    cos = np.sqrt(log) * np.cos(2 * np.pi * rnd2)
    sin = np.sqrt(log) * np.sin(2 * np.pi * rnd2)
    
    jrnd = javg + (jdelta * cos)
    #y = javg + (jdelta * sin)     this might be forgotten and not implemented 
    
    return jrnd       # Assuming it is to return (x, y)


   def therm(eh, energies_m, q_m, nmol):       
      return np.dot(np.exp(-eh * (energies_m / self.temp)), q_m / nmol)
   
   def thermQ(eh, energies_m, q_m, nmol):
    
    if eh == 1:
        idx = 0  
    elif eh == -1:                            
        idx = nmol-1
    else:
        msg = 'isw_H must be 1 or -1'
        raise NotImplementedError(msg)    

    return np.dot(np.exp(-eh * ((energies_m - energies_m[idx]) / self.temp)) ,  q_m / nmol)