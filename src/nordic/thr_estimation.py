import numpy as np

def thr_est_mppca(S: np.ndarray, M: int, N: int, demeaned: bool = True, return_noise: bool = False):
    """
    Estimates the threshold for denoising using the MPPCA method.

    Args:
        S (ndarray): Singular values from SVD.
        M (int): Number of principal components to retain.
        N (int): Number of samples in the input signal.
        demeaned (bool, optional): Flag indicating whether the input signal is demeaned. Defaults to True.
        return_noise (bool, optional): Flag indicating whether to return the estimated noise. Defaults to False.

    Returns:
        Union[ndarray, Tuple[ndarray, ndarray]]: Denoised signal or tuple of denoised signal and estimated noise.

    Raises:
        ValueError: If M is not larger than N.

    Examples:
        >>> S = np.array([1, 2, 3, 4, 5])
        >>> thr_est_mppca(S, 2, 5)
        array([3, 4])
    """
    if M <= N:
        raise ValueError("M must be larger than N")
    centering = 1 if demeaned else 0
    
    vals = (S**2)/N
    gamma = (M-np.array(range(N-centering)))/N
    
    csum = np.cumsum(vals[N-centering-1:None:-1])
    cmean = csum[N-centering-1:None:-1]/range(N-centering, 0, -1)
    sigmasq_1 = cmean / gamma
    
    rangeMP = 4*np.sqrt(gamma[:])
    rangeData = vals[:N-centering]-vals[N-centering-1]
    sigmasq_2 = rangeData/rangeMP
    t = np.where(sigmasq_2 < sigmasq_1)
    thr_idx = t[0][0]
    
    return (S[thr_idx], sigmasq_2[thr_idx]) if return_noise else S[thr_idx]


def thr_est_nordic(noise, M, N, scale=1, niter=100):
    """
    threshold estimation proposed by NORDIC denoising method

    Parameters:
        noise: noise signal
        M(int): number of voxels, M must larger than N
        N(int): number of frames, correspond to R (rank)
        scale: scaling factor for noise threshold

    Returns:
        sigma: noise level (threshold level of singular value)
    """
    noise_std = np.std(np.real(noise[noise != 0]))

    sigma = 0
    # Monte-Carlo simulation
    for _ in range(niter):
        sim_noise = (np.random.randn(M, N) + 1j*np.random.randn(M, N))
        s = np.linalg.svd(sim_noise, full_matrices=False)[1]
        sigma += np.abs(s).max()

    return (sigma / niter) * scale * noise_std