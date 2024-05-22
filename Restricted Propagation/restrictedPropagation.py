import numpy as np
import feret
from skimage.segmentation import morphological_geodesic_active_contour, morphological_chan_vese
from autoSegment import autoSegment, denoise_slice

def get_max_caliper_diameter(mask):
    """ Compute the maximum caliper diameter of a binary mask.
    Inputs
    ----------
    mask : 2D array (N,M)
        Binary mask.
    Returns
    -------
    max_caliper_diameter : Integer
        Maximum caliper diameter of the mask.
    """
    num_px = np.sum(mask == 1)
    if num_px <= 1:
        return 0
    else:
        max_caliper_diameter = feret.max(mask)
    return max_caliper_diameter

def propagate_mask_restricted(current_slice, current_mask, slice_set, denoised_method, active_contours_method):
    """ Propagate a mask from a given slice to the rest of the volume.
    Inputs
    ----------
    current_slice : Integer
        Index of the current slice.
    current_mask : 2D array (N,M)
        Mask to be propagated.
    slice_set : 3D array (N,M,L)
        Set of slices (Volume) to propagate the mask.
    denoised_method : String
        "tvl1": Total variation L1 denoising.
        "median": Median filter.
    active_contours_method : String
        "acwe": Chan-Vese active contours without edges.
        "geodesic": Geodesic active contours.
    Returns
    -------
    propagated_masks : 3D array (N,M,L)
        Set of propagated masks (Volume).
    """
    limits = [0, slice_set.shape[2]-1]
    limits_up = [current_slice + 1, limits[1]]
    limits_down = [limits[0], current_slice - 1]

    propagated_masks = np.zeros((current_mask.shape[0], current_mask.shape[1], slice_set.shape[2]))
    propagated_masks[:,:,current_slice] = current_mask
    max_length = get_max_caliper_diameter(current_mask)
    # Loop from current to last slice
    prev_mask = current_mask
    for i in range(limits_up[0], limits_up[1] + 1):
        
        I = slice_set[:, :, i]
        I = (I - np.min(I)) / (np.max(I) - np.min(I))
        J = denoise_slice(I, denoised_method)
        J = (J - np.min(J)) / (np.max(J) - np.min(J))
    
        if active_contours_method == 'acwe':
            ls = morphological_chan_vese(J, num_iter=10, lambda1=1, lambda2=1, init_level_set=prev_mask, smoothing=4)
        if active_contours_method == 'geodesic':
            ls = morphological_geodesic_active_contour(J, num_iter=10, init_level_set=prev_mask, smoothing=4)

        current_length = get_max_caliper_diameter(ls)
        
        # Ensure new contour does not expand beyond the initial mask
        if current_length > max_length:
            ls = prev_mask
        
        propagated_masks[:, :, i] = ls
    
        prev_mask = ls

    # Loop from current to first slice
    prev_mask = current_mask 
    for i in range(limits_down[1], limits_down[0]-1, -1):
        
        I = slice_set[:, :, i]
        I = (I - np.min(I)) / (np.max(I) - np.min(I))
        J = denoise_slice(I, denoised_method)
        J = (J - np.min(J)) / (np.max(J) - np.min(J))
    
        if active_contours_method == 'acwe':
            ls = morphological_chan_vese(J, num_iter=10, lambda1=1, lambda2=1, init_level_set=prev_mask, smoothing=4)
        if active_contours_method == 'geodesic':
            ls = morphological_geodesic_active_contour(J, num_iter=10, init_level_set=prev_mask, smoothing=4)

        current_length = get_max_caliper_diameter(ls)
        
        # Ensure new contour does not expand beyond the initial mask
        if current_length > max_length:
            ls = prev_mask
    
        propagated_masks[:, :, i] = ls
    
        prev_mask = ls

    return propagated_masks

def restrictedPropagation(tensor_4d, t_slice, dimension, denoised_method, active_contours_method):
    """ Propagate a mask from a given slice to the rest of the volume.
    Inputs
    ----------
    tensor_4d : 4D array (N,M,L,T)
        4D tensor to propagate the mask.
    t_slice : Integer
        Index of the current slice.
    dimension : Integer
        Dimension to propagate the mask.
    denoised_method : String
        "tvl1": Total variation L1 denoising.
        "median": Median filter.
    active_contours_method : String
        "acwe": Chan-Vese active contours without edges.
        "geodesic": Geodesic active contours.
    Returns
    -------
    propagated_masks : 3D array (N,M,L)
        Set of propagated masks (Volume).
    """
    central_indx = tensor_4d.shape[dimension] // 2

    if dimension == 0:
        slice = tensor_4d[central_indx, :, :, t_slice]
    elif dimension == 1:
        slice = tensor_4d[:, central_indx, :, t_slice]
    elif dimension == 2:
        slice = tensor_4d[:, :, central_indx, t_slice]
    else:
        raise ValueError("Invalid dimension value")

    central_mask = autoSegment(slice, denoised_method, active_contours_method)

    volume = tensor_4d[:,:,:,t_slice]
    slice_set = np.swapaxes(volume, dimension, 2)

    propagated_masks = propagate_mask_restricted(central_indx, central_mask, slice_set, denoised_method, active_contours_method)
    
    return propagated_masks