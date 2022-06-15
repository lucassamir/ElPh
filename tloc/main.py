import json
from pathlib import Path
from tloc.javerage import javerage
from tloc.jdelta import jdelta
from tloc.visualization import view
from tloc.molecules import Molecules

def write_lattice_file():
   """Write the lattice parameters json file
   """
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
   params = {'javg':[0.058, 0.058, 0.058],
             'jdelta':[0.029, 0.029, 0.029],
             'nrepeat':50,
             "iseed":3987187,
             'invtau':0.005,
             'temp':0.025
   }
   with open('params.json', 'w', encoding='utf-8') as f:
      json.dump(params, f, ensure_ascii=False, indent=4)

def main(args=None):
   import argparse

   description = "Transient Localization Theory command line interface"

   example_text = """
   examples:

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
      nmuc: 
      coordmol: 
      unitcell: 
      supercell: 
      unique:
      uniqinter: 

   """
   parser.add_argument('--lattice_file', nargs='*', help=help,
                        default='lattice', type=str)

   help = """
   All calculations require a params json 
   file with the following properties:

   params.json:
      javg: 
      jdelta: 
      nrepeat:
      iseed: 
      invtau:
      temp:

   """
   parser.add_argument('--params_file', nargs='*', help=help,
                        default='params', type=str)

   help = ("write example of lattice and params files")
   parser.add_argument('--write_files', action='store_true' , help=help)

   help = """
   Calculate average transfer integral 
   for all unique pairs of molecules

   Gaussian calculator is required. Define the path with:
   ASE_GAUSSIAN_COMMAND='g09 < PREFIX.com > PREFIX.log'

   """
   parser.add_argument('--javerage', action='store_true' , help=help)

   help = """
   Calculate variance of the transfer integral 
   for all unique pairs of molecules

   Javerage step and phonon modes (mesh.yaml) are required

   """
   parser.add_argument('--jdelta', action='store_true' , help=help)

   help = """
   Visualization tool of sigma

   [atoms] View the total sigma contribution for each atom
   [modes] [[n]] View the n phonon modes with the highest sigma contribution

   """
   parser.add_argument('--view', nargs=2, default=['atoms', 1], type=str, help=help)

   help = ("Calculate charge mobility")
   parser.add_argument('--mobility', action='store_true' , help=help)

   args = parser.parse_args(args)

   if args.write_files:
      write_lattice_file()
      write_params_file()
      return

   print('Initializing TLOC')

   if args.javerage:
      javerage()

   if args.jdelta:
      jdelta()

   if args.view:
      view(*args.view)

   if args.mobility:
      if not Path(args.lattice_file + '.json').is_file():
         msg = 'Lattice file could not be found'
         raise FileNotFoundError(msg)

      if not Path(args.params_file + '.json').is_file():
         msg = 'Params file could not be found'
         raise FileNotFoundError(msg)

      mols = Molecules(lattice_file=args.lattice_file, 
      params_file=args.params_file)
         
      mobx, moby = mols.get_mobility()
      with open('results.json', 'w', encoding='utf-8') as f:
         json.dump(mols.results, f)

if __name__  == '__main__':
   main()
