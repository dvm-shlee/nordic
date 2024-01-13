from .patch_utils import *
from .thr_estimation import thr_est_mppca


def prep_llr_process(input_data, gamma, overlap_factor):
    data = input_data.copy()
    # normalize scale
    abs_scale = data[data != 0].min()
    data /= abs_scale

    if isinstance(gamma, list) or isinstance(gamma, tuple):
        patch_dim = gamma
    else:
        patch_dim = get_patch_dim(data, gamma)
    patch_coords = get_patch_coords(data, patch_dim, overlap_factor)
    return data, abs_scale, patch_dim, patch_coords

def noise_estimation(input_data, demean: bool = True):
    """ extimate gfactor (sensativity map) using MPPCA
    """
    data = input_data.copy()
    m, n = data.shape
    
    if demean:
        colmean = np.tile(data.mean(1)[:, np.newaxis], (1, n))
        data -= colmean

    u, s, v = np.linalg.svd(data, full_matrices=False)
    _, noise = thr_est_mppca(s, m, n, demeaned=demean, return_noise=True)
    return noise


def gfactor_estimation(input_data, gamma, overlap_factor):
    data = input_data.copy() 
    data, abs_scale, patch_dim, patch_coords = prep_llr_process(data, gamma, overlap_factor)
    data = data[..., :min(90, data.shape[3])]
    gfactor = np.zeros(data.shape[:3])
    weight = gfactor.copy().astype(int)

    for patch_coord in patch_coords:
        cmat = get_casorati_matrix(data, patch_coord, patch_dim)
        noise = noise_estimation(cmat, demean=False)
        slicers = get_patch_slicers(patch_coord, patch_dim)
        gfactor[slicers] = gfactor[slicers] + noise
        weight[slicers] += 1

    return np.sqrt(gfactor / weight) * abs_scale