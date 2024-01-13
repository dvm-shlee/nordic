import numpy as np
from .patch_utils import get_2d_slicer
from scipy.signal import get_window
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from typing import Tuple, List, Optional


def get_2d_slicer(data_shape: Tuple[int, int, int, Optional[int]], 
                  slice_axis: int, slice_idx: int, frame_idx: Optional[int]) -> Tuple[slice, ...]:
    """
    Generates a slicer for extracting a 2D slice from 3D or 4D data.

    Args:
        data_shape (Tuple[int, int, int, Optional[int]]): Shape of the input data.
        slice_axis (int): Axis along which to extract the slice.
        slice_idx (int): Index of the slice along the specified axis.
        frame_idx (Optional[int], optional): Index of the frame. Defaults to None.

    Returns:
        Tuple[slice, ...]: Slicer for extracting the 2D slice.

    Examples:
        >>> data_shape = (100, 100, 100)
        >>> slice_axis = 1
        >>> s = 5
        >>> get_2d_slicer(data_shape, slice_axis, slice_idx)
        (slice(None), 5, slice(None))

        >>> data_shape = (100, 100, 100, 3)
        >>> slice_axis = 1
        >>> s = 5
        >>> n = 2
        >>> get_2d_slicer(data_shape, slice_axis, s, n)
        (slice(None), 5, slice(None), 2)
    """
    slicer = [slice(None)] * len(data_shape)
    slicer[slice_axis] = slice_idx
    slicer[-1] = frame_idx
    return tuple(slicer)


def select_slice_axis(data_shape: tuple[int, int, int, int]) -> int:
    """
    Selects the slice axis based on the shape of the data.

    Args:
        data_shape: Shape of the data.

    Returns:
        The selected slice axis.
    """
    return 2 if all(data_shape[:3]) else np.argmin(data_shape[:3])


def cal_2d_phase(data: np.ndarray, phase_filter_width: int, temporal_phase: int) -> np.ndarray:
    """
    Calculates the phase from the given 2D data.

    Args:
        data (ndarray): Input 2D data.
        phase_filter_width (int): Width of the phase filter.
        temporal_phase (int): Temporal phase correction mode.

    Returns:
        ndarray: Phase calculated from the 2D data.

    Raises:
        ValueError: If the temporal_phase value is invalid.

    Examples:
        >>> data = np.zeros((100, 100))
        >>> cal_2d_phase(data, phase_filter_width=1, temporal_phase=2)
        array([...])
    """
    phase = data.copy()
    for ndim in [0, 1]:
        phase = ifftshift(ifftn(ifftshift(phase, axes=(ndim,)), axes=(ndim,)))
        phase *= (get_window(('tukey', 1), phase.shape[ndim], fftbins=False) ** phase_filter_width).reshape(
            [-1 if dim == ndim else 1 for dim in range(phase.ndim)]
        )
        phase = fftshift(fftn(fftshift(phase, axes=(ndim,)), axes=(ndim,)))

    if temporal_phase > 1:
        phase_diff = np.angle(phase) / phase
        mask = np.abs(phase_diff) > 1
        if temporal_phase == 2:
            phase[mask] = data[mask]
        elif temporal_phase == 3:
            mask &= (np.abs(data) > np.sqrt(2))
            phase[mask] = data[mask]
        else:
            raise ValueError("Invalid temporal_phase value")
    return phase


def phase_stabilization(input_data: np.ndarray,
                        temporal_phase: int,
                        phase_filter_width: int,
                        slice_axis: int = None,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs (x+t) phase stabilization[1] on the input data.

    Args:
        input_data (ndarray): Input data.
        temporal_phase (int): Temporal phase correction mode.
        phase_filter_width (int): Width of the phase filter.
        slice_axis (int, optional): Axis along which to perform slice-wise phase correction. Defaults to None.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: Tuple containing the stabilized data, mean phase, and slice-wise phase differences.

    Examples:
        >>> input_data = np.zeros((100, 100, 100, 3))
        >>> temporal_phase = 2
        >>> phase_filter_width = 1
        >>> phase_stabilization(input_data, temporal_phase, phase_filter_width)
        (array([...]), array([...]), array([...]))
        
    References:
        1. Moeller et al., NeuroImage 226 (2021) 117539
    """
    data = input_data.copy()
    mean_phase = data.mean(-1)
    data *= np.exp(-1j * np.angle(np.tile(mean_phase[..., np.newaxis], (1, 1, 1, data.shape[-1]))))

    if temporal_phase > 0:
        slice_wise_phase_diff = np.zeros_like(data)
        slice_axis = slice_axis or select_slice_axis(data.shape)
        for s in range(data.shape[slice_axis]):
            for n in range(data.shape[-1]):
                slicer = get_2d_slicer(data.shape, slice_axis, s, n)
                sliced_data = data[slicer]
                sliced_phase = cal_2d_phase(sliced_data, phase_filter_width, temporal_phase)
                slice_wise_phase_diff[slicer] = sliced_phase

        data *= np.exp(-1j * np.angle(slice_wise_phase_diff))
    return data, mean_phase, slice_wise_phase_diff

