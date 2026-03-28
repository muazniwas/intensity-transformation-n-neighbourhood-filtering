import numpy as np


def bilateral_filter(img, diameter, sigma_s, sigma_r):
    """
    Manual bilateral filter for grayscale images.

    Each output pixel is a weighted average of its neighbours, where the
    weight combines:
      - Spatial Gaussian  : penalises neighbours far away in position
      - Range Gaussian    : penalises neighbours far away in intensity

        W(i,j,k,l) = exp(-(dist_space^2)/(2*sigma_s^2))
                    * exp(-(dist_range^2)/(2*sigma_r^2))

    Parameters
    ----------
    img      : 2D uint8 grayscale image
    diameter : kernel diameter (must be odd; if even, it is incremented by 1)
    sigma_s  : spatial standard deviation (controls spatial falloff)
    sigma_r  : range standard deviation  (controls intensity falloff)

    Returns
    -------
    Filtered image as uint8.
    """
    if diameter % 2 == 0:
        diameter += 1

    radius = diameter // 2
    img_f  = img.astype(np.float64)
    H, W   = img.shape

    # Pre-compute spatial Gaussian weights for the kernel window
    ax = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(ax, ax)
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    # Pad image to handle borders
    padded = np.pad(img_f, radius, mode='reflect')
    output = np.zeros_like(img_f)

    for i in range(H):
        for j in range(W):
            neighbourhood = padded[i:i + diameter, j:j + diameter]
            center        = img_f[i, j]

            range_weights = np.exp(-(neighbourhood - center)**2 / (2 * sigma_r**2))
            weights       = spatial_weights * range_weights

            output[i, j]  = np.sum(weights * neighbourhood) / np.sum(weights)

    return output.astype(np.uint8)
