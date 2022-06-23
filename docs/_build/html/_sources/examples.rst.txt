Examples
====================================

Example 1: Workflow on local machine
^^^^^^^^^^^^^^^^^^^^^

The following example shows the complete workflow run on a local machine.

* Calculate transfer integral between pairs of molecules (J :sub:`average`):

First, create a folder containing the geometry file (.cif, .gen, .sdf, or .xyz). 
The folder used in this example, named Anthracene, can be downloaded from the Uploads Folder.

In the Anthracene folder, unwrap the structure to whole molecules, 
and calculate the transfer integral between each unique pair of molecules in the system, 
using the following command.

.. code-block::

   elph --javerage

Once the job has completed, the following files and folders can be found in the Anthracene folder.

.. code-block::

   1/    A/    950158.cif        atom_mapping.json    J_C.json
   2/    B/    all_pairs.json    J_A.json
   3/    C/    all_pairs.xyz     J_B.json

The J files (J_A.json, J_B.json, J_C.json) present the transfer integral in meV
of each pair described in all_pairs.json.

* Calculate the variance of transfer integrals (Sigma):

Before calculating Sigma, which is the variance of the transfer integral due to vibrations in the system,
the phonons have to be precomputed. `DCS-Flow <https://dcs-flow.readthedocs.io/en/master/index.html>`_ 
calculates the phonon modes as the second part of its own workflow (2-phonons). 

Copy the following files to the Anthracene folder

.. code-block::

   FORCE_SETS    phonopy_params.yaml

Calculate the variance (Sigma) within the finite differences method using the command
  
.. code-block::

   elph --sigma

After the job is done, the following files and folders will be written in the Anthracene folder.

.. code-block::

   1/displacements/...    A/displacements/...    A_disp_js.npz    Sigma_A.json    phonon.npz
   2/displacements/...    B/displacements/...    B_disp_js.npz    Sigma_B.json
   3/displacements/...    C/displacements/...    C_disp_js.npz    Sigma_A.json

The Sigma files (Sigma_A.json, Sigma_B.json, Sigma_C.json) present the variance of the transfer integral 
in meV of each pair

* Calculate the mobility

Create the lattice and parameters files, ``lattice.json``\ and ``params.json``\, with the command

.. code-block::

   elph --write_files

Edit the files to match the following values

lattice.json: 

.. code-block::
   {
      "nmuc": 2,
      "coordmol": [
         [0.0, 0.0, 0.0],
         [0.5, 0.5, 0.0]
         ],
      "unitcell": [
         [1.0, 0.0, 0.0],
         [0.0, 1.7321, 0.0],
         [0.0, 0.0, 1000.0]
      ],
      "supercell": [16, 16, 1],
      "unique": 6,
      "uniqinter": [
         [1, 1, 1, 0, 0, 1],
         [2, 2, 1, 0, 0, 1],
         [1, 2, 0, 0, 0, 3],
         [2, 1, 1, 0, 0, 2],
         [2, 1, 0, 1, 0, 2],
         [2, 1, 1, 1, 0, 3]
      ]
   }

params.json: 

.. code-block::
   {
      "javg": [0.058, 0.058, 0.058],
      "sigma": [0.029, 0.029, 0.029],
      "nrepeat": 50,
      "iseed": 3987187,
      "invtau": 0.005,
      "temp": 0.025
   }

Use the following command to calculate the mobility (in cm:sub:`2`/(V . s))

.. code-block::
   elph --mobility

* Visualize Sigma

In order to visualize the atomic contributions to Sigma, run

.. code-block::
   elph --view atoms

Or to visualize the 3 highest contributing phonon modes to Sigma, used

.. code-block::
   elph --view modes 3