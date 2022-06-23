Install
====================================

ElPh Requirements
^^^^^^^^^^^^^^^^^^^^^


* `Gaussian <https://gaussian.com/>`_
* `Catnip <https://hub.docker.com/r/madettmann/catnip>`_

Installation
^^^^^^^^^^^^^^^^^^^^^


* Install ElPh:

.. code-block::

   pip install elph


* Install gaussian:

  #. 
     Gaussian installation will vary group to group.

  
* Installing Catnip
    
    Download and install `Docker <https://docs.docker.com/get-docker/>`_

.. code-block::

   docker pull madettmann/catnip



Set environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^

Add these lines to your configuration file (.bashrc). The following code uses example paths and must be edited according to your system.

.. code-block::

   export ELPH_CATNIP_CMD='docker run -i --rm -v $(pwd):/projects -u $(id -u):$(id -g) madettmann/catnip'
   export ASE_GAUSSIAN_COMMAND='Your Gaussian Command Here < PREFIX.com > PREFIX.log'


Usage
^^^^^



#. 
   Add cif file to current working directory

   
#. 
   Run javerage

    .. code-block::

        elph --javerage

#. 
   Compute the phonon modes

   Refer to `DCS-Flow <https://dcs-flow.readthedocs.io/en/master/index.html>`_ for more information.
    
#.
   Run sigma

   Copy FORCE_SETS, phonopy_params.yaml to your working directory

   .. code-block::

       elph --sigma

#. 
   Compute mobility

   Write input files and define correct parameters for the specific system

   .. code-block::

       elph --write_files

   Run mobility

   .. code-block::

       elph --mobility

#. 
   Visualize sigma contribution per atom/mode

   To generate the visualization per atom:

   .. code-block::

       elph --view atoms

   To generate the visualization per mode (n highest modes):

   .. code-block::

       elph --view modes 3