import numpy as np
import skimage
from skimage.segmentation import disk_level_set, morphological_geodesic_active_contour, morphological_chan_vese

def denoise_slice(slice, method):
    """ Preprocessing of images with different methods.
    Inputs
    ----------
    slice : 2D array (N,M)
        Image slice to be denoised.
    method : String
        "tvl1": Total variation L1 denoising.
        "median": Median filter.

    Returns
    -------
    J : 2D array (N,M)
        Denoised image slice.
    """
    if np.max(slice) > 1:
        slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))

    if method == 'tvl1':
        J = solve_TVL1(slice, 0.5, iter_n=100)
    if method == 'median':
        J = skimage.filters.median(slice, skimage.morphology.disk(5))
    J = -J
    return J

def autoSegment(slice, denoised_method, active_contours_method):
    """ Automatic segmentation of a 2D image slice.
    Inputs
    ----------
    slice : 2D array (N,M)
        Image slice to be segmented.
    denoised_method : String
        "tvl1": Total variation L1 denoising.
        "median": Median filter.
    active_contours_method : String
        "acwe": Chan-Vese active contours without edges.
        "geodesic": Geodesic active contours.
    Returns
    -------
    mask : 2D array (N,M)
        Binary mask of the segmented image slice.
    """
    # Pre-processing
    I = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
    J = denoise_slice(I, denoised_method)
    denoised = (J - np.min(J)) / (np.max(J) - np.min(J))
    ## 1. FIND BLOBS
    min_sigma = 1
    max_radius = min(denoised.shape) // 2
    max_sigma = max_radius
    threshold = 0.1

    num_blobs = float('inf')  # Start with infinity blobs
    sigma = min_sigma
    found_blob = None
    
    # Perform blob detection
    while num_blobs > 1 and sigma <= max_sigma:
        # Detect blobs
        blobs_log = skimage.feature.blob_log(denoised, min_sigma=sigma, max_sigma=sigma, num_sigma=1, threshold=threshold)
    
        # Filter out blobs whose circles fit inside the image
        valid_blobs = []
        for blob in blobs_log:
            y, x, r = blob
            if (x - r >= 0) and (x + r < denoised.shape[1]) and (y - r >= 0) and (y + r < denoised.shape[0]):
                valid_blobs.append(blob)
    
        # Update number of blobs
        num_blobs = len(valid_blobs)

        # If only one valid blob is found, store its coordinates and break the loop
        if num_blobs == 1:
            found_blob = valid_blobs[0]
            break
    
        # Increment sigma
        sigma += 1
    if found_blob is not None:
        init_contour = disk_level_set(image_shape=denoised.shape, center=(found_blob[0], found_blob[1]), radius=found_blob[2])
        if active_contours_method == 'acwe':
            mask = morphological_chan_vese(denoised, num_iter=60, lambda1=1, lambda2=1, init_level_set=init_contour,smoothing=2)
        if active_contours_method == 'geodesic':
            mask = morphological_geodesic_active_contour(denoised, num_iter=100, init_level_set=init_contour, smoothing=2, balloon=1)
    else:
        mask = np.zeros_like(denoised)

    return mask


############################################################################################################
# The following functions for TV-L1 denoising were taken from: https://github.com/znah/notebooks/tree/master
def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def nabla(I):
    h, w = I.shape
    G = np.zeros((h, w, 2), I.dtype)
    G[:, :-1, 0] -= I[:, :-1]
    G[:, :-1, 0] += I[:, 1:]
    G[:-1, :, 1] -= I[:-1]
    G[:-1, :, 1] += I[1:]
    return G

def nablaT(G):
    h, w = G.shape[:2]
    I = np.zeros((h, w), G.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    I[:, :-1] -= G[:, :-1, 0]
    I[:, 1: ] += G[:, :-1, 0]
    I[:-1]    -= G[:-1, :, 1]
    I[1: ]    += G[:-1, :, 1]
    return I

def project_nd(P, r):
    '''perform a pixel-wise projection onto R-radius balls'''
    nP = np.maximum(1.0, anorm(P)/r)
    return P / nP[...,np.newaxis]
    
def shrink_1d(X, F, step):
    '''pixel-wise scalar srinking'''
    return X + np.clip(F - X, -step, step)

def calc_energy_TVL1(X, observation, clambda):
    Ereg = anorm(nabla(X)).sum()
    Edata = clambda * np.abs(X - observation).sum()
    return Ereg + Edata

def solve_TVL1(img, clambda, iter_n=100):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    X = img.copy()
    P = nabla(X)
    for i in range(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        X1 = shrink_1d(X - tau*nablaT(P), img, clambda*tau)
        X = X1 + theta * (X1 - X)
        #if i % 10 == 0:
        #    print("%.2f" % calc_energy_TVL1(X, img, clambda), end=" ")
    #print()
    return X
############################################################################################################