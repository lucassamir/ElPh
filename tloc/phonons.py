import phonopy
import numpy as np

def get_phase_factor(modulation):
    u = np.ravel(modulation)
    index_max_elem = np.argmax(abs(u))
    max_elem = u[index_max_elem]
    phase_for_zero = max_elem / abs(max_elem)
    phase_factor = np.exp(1j * np.pi / 180) / phase_for_zero

    return phase_factor

def get_displacements(spod, q, masses, eigvec):
    m = masses
    coefs = np.exp(2j * np.pi * np.dot(spod, q)) / np.sqrt(m)
    u = []
    for i, coef in enumerate(coefs):
        eig_index = i * 3
        u.append(eigvec[eig_index : eig_index + 3] * coef)

    u = np.array(u) / np.sqrt(len(m))
    #phase_factor = get_phase_factor(u)
    #u *= phase_factor

    return u

def write_phonons(mesh=[8, 8, 8], phonopy_file="phonopy_params.yaml"):
    phonon = phonopy.load(phonopy_file)
    phonon.run_mesh(mesh, with_eigenvectors=True)
    pmesh = phonon.get_mesh_dict()
    
    # nqpoints x nbands from phonopy to e modes
    freqs_e = pmesh['frequencies'].ravel()

    # converting energy unit
    thz2ev = 4.13566733e-3 
    freqs_e *= thz2ev # eV

    # nqpoints
    nq = len(pmesh['qpoints'])

    # masses, frac_coords
    masses = phonon._mesh._cell.masses
    spod = phonon._mesh._cell.get_scaled_positions()

    # transform eigenvectors to eigendisplacements
    phonon_modes = []
    for i, q in enumerate(pmesh['qpoints']):
        for j in range(len(pmesh['frequencies'][0])):
            phonon_modes.append([i, q, j])

    u = np.zeros((len(phonon_modes), len(masses), 3),
                 dtype=np.complex128,
                 order="C",)

    for i, ph_mode in enumerate(phonon_modes):
            q_index, q, band_index = ph_mode
            disp = get_displacements(spod, q, masses, pmesh['eigenvectors'][q_index, :, band_index])
            u[i] = disp
            
    # avoid negative (imaginary) frequencies
    #ind = np.where(freqs_e > 0)
    #freqs_e = freqs_e[ind]
    #vecs_eav = vecs_eav[ind]

    data = {'freqs': freqs_e,
            'vecs': u.real,
            'nq': nq}
    np.savez_compressed('phonon.npz', **data)