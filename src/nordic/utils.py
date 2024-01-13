from typing import Tuple, List
import numpy as np


def absolute_scale(input_data):
    """
    Normalizes the scale of the input data by dividing it with the minimum non-zero value.

    Args:
        input_data (ndarray): The input data to be scaled.

    Returns:
        tuple: A tuple containing the scaled data and the absolute scale.

    Examples:
        >>> input_data = np.random.rand(100, 100)
        >>> absolute_scale(input_data)
        (array([...]), 0.123)
    """
    data = input_data.copy()
    # normalize scale
    abs_scale = data[data != 0].min()
    data /= abs_scale
    return data, abs_scale


def get_patch_coords(data: np.ndarray, 
                     patch_dim: Tuple[int, int, int], 
                     overlap_factor: int) -> List[Tuple[int, int, int]]:
    """
    Generates coordinates for extracting patches from data.

    Args:
        data (ndarray): Input data.
        patch_dim (Tuple[int, int, int]): Dimensions of the patch.
        overlap_factor (int): Overlap factor for the patches.

    Returns:
        List[Tuple[int, int, int]]: Coordinates of the patches.

    Examples:
        >>> data = np.zeros((100, 100, 100))
        >>> patch_dim = (10, 10, 10)
        >>> overlap_factor = 2
        >>> get_patch_coords(data, patch_dim, overlap_factor)
        [(0, 0, 0), (0, 0, 5), (0, 0, 10), ...]
    """
    dim = np.array(data.shape[:3])
    patch_dim = np.array(patch_dim)
    step_size = np.floor(patch_dim/overlap_factor).astype(int) # step size
    num_steps = np.floor(dim/step_size).astype(int)

    # add one more step to prevent incompletion
    if not np.all((dim - (num_steps * step_size)) == 0):
        num_steps += 1
    
    patch_coords = []
    for x in np.linspace(0, dim[0]-patch_dim[0], num_steps[0]).astype(int):
        for y in np.linspace(0, dim[1]-patch_dim[1], num_steps[1]).astype(int):
            patch_coords.extend(
                (x, y, z)
                for z in np.linspace(
                    0, dim[2] - patch_dim[2], num_steps[2]
                ).astype(int)
            )
    return patch_coords


def get_patch_dim(data: np.ndarray, gamma: int = 11) -> List[int]:
    """
    Calculate the dimensions of a patch from image data based on a target ratio.

    Parameters:
        data (ndarray): Image data.
        gamma (int, optional): Target ratio for the Casorati matrix. Defaults to 11.

    Returns:
        list: Dimensions of the patch.

    Raises:
        ValueError: If the given gamma is too big for the image dimension.

    Example:
        >>> data = np.zeros((100, 100, 100, 3))
        >>> get_patch_dim(data, gamma=10)
        [10, 10, 10]
    """
    dx, dy, dz, df = data.shape
    ref_size = df * gamma
    patch_size = np.round(ref_size ** (1/3)).astype(int)
    dim = np.array([dx, dy, dz])
    
    small_axes = np.where(dim < patch_size)[0]
    if small_axes.shape[0] == 0:
        patch_dim = [patch_size] * 3
    else:
        patch_dim = [0, 0, 0]
        
        if small_axes.shape[0] == 1:
            aid = small_axes[0]
            ref_size /= dim[aid]
            for i in range(3):
                if i == aid:
                    patch_dim[i] = dim[i]
                else:
                    patch_size = np.round(np.sqrt(ref_size)).astype(int)
                    if patch_size > np.round(dim[i]/2).astype(int):
                        raise ValueError('given gamma is too big than image dimension')
                    else:
                        patch_dim[i] = patch_size
        
        if small_axes.shape[0] > 1:
            raise ValueError('given gamma is too big than image dimension')
    return patch_dim



def get_patch_slicers(coord: Tuple[int, int, int],
                      patch_dim: Tuple[int, int, int]) -> Tuple[slice, ...]:
    """
    Generates slicers for extracting a patch from data.

    Args:
        coord (Tuple[int, int, int]): Coordinates of the patch.
        patch_dim (Tuple[int, int, int]): Dimensions of the patch.

    Returns:
        Tuple[slice, slice, slice]: Slicers for extracting the patch.

    Examples:
        >>> coord = (0, 0, 0)
        >>> patch_dim = (2, 2, 2)
        >>> get_slicers(coord, patch_dim)
        (slice(0, 2, None), slice(0, 2, None), slice(0, 2, None))
    """
    x, y, z = coord
    px, py, pz = patch_dim
    
    xs = slice(x, x+px)
    ys = slice(y, y+py)
    zs = slice(z, z+pz)
    return xs, ys, zs


def get_casorati_matrix(data: np.ndarray, 
                        coord: Tuple[int, int, int], 
                        patch_dim: Tuple[int, int, int]) -> np.ndarray:
    """
    Retrieves the Casorati matrix from the given data based on the specified coordinates and patch dimensions.

    Args:
        data (ndarray): Input data.
        coord (Tuple[int, int, int]): Coordinates of the patch.
        patch_dim (Tuple[int, int, int]): Dimensions of the patch.

    Returns:
        ndarray: Casorati matrix obtained from the data.

    Examples:
        >>> data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> coord = (0, 0, 0)
        >>> patch_dim = (2, 2, 2)
        >>> get_casorati_matrix(data, coord, patch_dim)
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
    """
    xs, ys, zs = get_patch_slicers(coord, patch_dim)
    return data[xs, ys, zs, :].reshape([np.prod(patch_dim), data.shape[-1]])


def prep_llr_process(input_data, gamma, overlap_factor):
    """
    Prepares the input data for the LLR (Local Low-Rank) process by normalizing the scale, determining the patch dimensions, and calculating the patch coordinates.

    Args:
        input_data (ndarray): The input data to be prepared.
        gamma (float or list or tuple): The regularization parameter or patch dimensions.
        overlap_factor (float): The overlap factor for patch extraction.

    Returns:
        tuple: A tuple containing the patch dimensions, and patch coordinates.

    Examples:
        >>> input_data = np.random.rand(100, 100)
        >>> gamma = 0.1
        >>> overlap_factor = 0.5
        >>> prep_llr_process(input_data, gamma, overlap_factor)
        (array([...]), 0.123, (8, 8), [(0, 0), (0, 8), ...])
    """
    data = input_data.copy()
    if isinstance(gamma, (list, tuple)):
        patch_dim = gamma
    else:
        patch_dim = get_patch_dim(data, gamma)
    patch_coords = get_patch_coords(data, patch_dim, overlap_factor)
    return patch_dim, patch_coords