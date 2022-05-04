import phonopy
import numpy as np

def write_phonons(mesh=[8, 8, 8], phonopy_file="phonopy_params.yaml"):
    phonon = phonopy.load(phonopy_file)
    phonon.run_mesh(mesh, with_eigenvectors=True)

    # nqpoints x nbands from phonopy
    freqs = phonon._mesh.frequencies

    # e modes 
    freqs_e = freqs.flatten()

    # converting energy unit
    thz2ev = 4.13566733e-3 / (2 * np.pi)
    freqs_e *= thz2ev # eV

    # nqpoints x (natoms x 3 directions) x nbands from phonopy
    vecs = phonon._mesh.eigenvectors
    
    # e modes, a atoms, v directions 
    vecs = np.transpose(vecs, axes=[0, 2, 1])
    vecs_eav = vecs.real.reshape(len(freqs_e), -1, 3)

    data = {'freqs': freqs_e,
            'vecs': vecs_eav}
    np.savez_compressed('phonon.npz', **data)