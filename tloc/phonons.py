import phonopy
import numpy as np

def write_phonons(mesh=[8, 8, 8], phonopy_file="phonopy_params.yaml"):
    phonon = phonopy.load(phonopy_file)
    phonon.run_mesh(mesh, with_eigenvectors=True)

    # nqpoints x nbands from phonopy
    freqs = phonon._mesh.frequencies

    # e modes 
    freqs_e = freqs.flatten()

    # masses
    masses = phonon._mesh._cell.masses

    # converting energy unit
    thz2ev = 4.13566733e-3 
    freqs_e *= thz2ev # eV

    # nqpoints x (natoms x 3 directions) x nbands from phonopy
    vecs = phonon._mesh.eigenvectors
    
    # e modes, a atoms, v directions 
    vecs = np.transpose(vecs, axes=[0, 2, 1])
    vecs_eav = np.absolute(vecs.reshape(len(freqs_e), -1, 3))

    # mass weighted
    vecs_eav /= np.sqrt(masses)[None, :, None]

    # avoid negative (imaginary) frequencies
    ind = np.where(freqs_e > 0)
    freqs_e = freqs_e[ind]
    vecs_eav = vecs_eav[ind]

    data = {'freqs': freqs_e,
            'vecs': vecs_eav}
    np.savez_compressed('phonon.npz', **data)