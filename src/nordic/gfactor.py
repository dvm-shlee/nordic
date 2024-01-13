import numpy as np
from .thr_estimation import thr_est_mppca
from .utils import get_patch_slicers, get_casorati_matrix, prep_llr_process


def noise_estimation(input_data, demean: bool = True):
    """
    Estimates the noise level in the input data using the MPPCA (Modified Principal Component Analysis) method.

    Args:
        input_data (ndarray): The input data for noise estimation.
        demean (bool, optional): Whether to demean the input data before estimation. Defaults to True.

    Returns:
        ndarray: The estimated noise level.

    Examples:
        >>> input_data = np.random.rand(100, 100)
        >>> noise_estimation(input_data)
        array([...])
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
    """
    Estimates the g-factor for a given input data using the LLR (Local Low-Rank) method.

    Args:
        input_data (ndarray): The input data for g-factor estimation.
        gamma (float or list or tuple): The regularization parameter or patch dimensions.
        overlap_factor (float): The overlap factor for patch extraction.

    Returns:
        ndarray: The estimated g-factor.

    Raises:
        ValueError: If the number of frames in the input data is less than 30.

    Examples:
        >>> input_data = np.random.rand(100, 100, 100, 10)
        >>> gamma = 0.1
        >>> overlap_factor = 0.5
        >>> gfactor_estimation(input_data, gamma, overlap_factor)
        array([...])
    """
    data = input_data.copy() 
    patch_dim, patch_coords = prep_llr_process(data, gamma, overlap_factor)
    
    if data.shape[3] < 30:
        raise ValueError("Number of frames must be greater than or equal to 30")
    data = data[..., :min(90, data.shape[3])]

    gfactor = np.zeros(data.shape[:3])
    weight = gfactor.copy().astype(int)

    for patch_coord in patch_coords:
        cmat = get_casorati_matrix(data, patch_coord, patch_dim)
        noise = noise_estimation(cmat, demean=False)
        slicers = get_patch_slicers(patch_coord, patch_dim)
        gfactor[slicers] = gfactor[slicers] + noise
        weight[slicers] += 1

    return np.sqrt(gfactor / weight)