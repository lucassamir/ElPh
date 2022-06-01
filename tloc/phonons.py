import phonopy
import numpy as np

def write_phonons(mesh=[8, 8, 8], phonopy_file="phonopy_params.yaml"):
    phonon = phonopy.load(phonopy_file)
    phonon.run_mesh(mesh, with_eigenvectors=True)
    
    # nqpoints x nbands from phonopy to e modes
    freqs_e = phonon.mesh.frequencies.flatten()

    # converting energy unit
    thz2ev = 4.13566733e-3 
    freqs_e *= thz2ev # eV

    # nqpoints x (natoms x 3 directions) x nbands from phonopy
    vecs = phonon.mesh.eigenvectors

    # masses, frac_coords and qpoints
    masses_a = phonon.mesh._cell.masses
    fcoords_av = phonon.mesh._cell.get_scaled_positions()
    qpoints_qv = phonon.mesh.qpoints
    nq = len(qpoints_qv)

    # transform eigenvectors to eigendisplacements (phonopy modulations)
    factor_qa = (np.exp(2j * np.pi * np.dot(fcoords_av, qpoints_qv.T)) / np.sqrt(masses_a)[:, None]).T
    vecs = np.repeat(factor_qa, 3).reshape(nq, 3 * len(masses_a))[:, :, None] * vecs
    # vecs /= np.sqrt(len(masses_a))
        
    # e modes, a atoms, v directions 
    vecs = np.transpose(vecs, axes=[0, 2, 1])
    vecs_eav = vecs.reshape(len(freqs_e), -1, 3)

    # add phase factor (phonopy modulations)
    vecs_e = vecs_eav.reshape(len(vecs_eav), -1)
    index_maxarr = np.abs(vecs_e).argmax(1)
    maxarr = np.take_along_axis(vecs_e, index_maxarr[:, None], axis=1)[:, 0]
    phase_for_zero = maxarr / np.abs(maxarr)
    phase_factor = np.exp(1j * np.pi / 180) / phase_for_zero
    vecs_eav *= phase_factor[:, None, None]

    # avoid negative (imaginary) frequencies
    ind = np.where(freqs_e > 0)
    freqs_e = freqs_e[ind]
    vecs_eav = vecs_eav[ind]

    data = {'freqs': freqs_e,
            'vecs': vecs_eav.real,
            'nq': nq}
    np.savez_compressed('phonon.npz', **data)