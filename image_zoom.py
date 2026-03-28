import numpy as np


def zoom(img, s, method='nearest'):
    """
    Zoom an image by scale factor s ∈ (0, 10].

    Parameters
    ----------
    img    : grayscale or color image (uint8)
    s      : scale factor, must be in (0, 10]
    method : 'nearest' for nearest-neighbor, 'bilinear' for bilinear

    Returns
    -------
    Zoomed image as uint8.
    """
    if not (0 < s <= 10):
        raise ValueError(f"Scale factor s must be in (0, 10], got {s}")

    h, w = img.shape[:2]
    new_h = int(round(h * s))
    new_w = int(round(w * s))

    if method == 'nearest':
        # Map each output pixel back to its nearest input pixel
        row_idx = np.floor(np.arange(new_h) / s).astype(int).clip(0, h - 1)
        col_idx = np.floor(np.arange(new_w) / s).astype(int).clip(0, w - 1)
        return img[np.ix_(row_idx, col_idx)]

    elif method == 'bilinear':
        # Map each output pixel to a continuous input coordinate
        row_src = np.arange(new_h) / s          # continuous row coords in input
        col_src = np.arange(new_w) / s          # continuous col coords in input

        row_src = np.clip(row_src, 0, h - 1)
        col_src = np.clip(col_src, 0, w - 1)

        r0 = np.floor(row_src).astype(int)
        c0 = np.floor(col_src).astype(int)
        r1 = np.clip(r0 + 1, 0, h - 1)
        c1 = np.clip(c0 + 1, 0, w - 1)

        dr = (row_src - r0)[:, np.newaxis]      # fractional row distance
        dc = (col_src - c0)[np.newaxis, :]      # fractional col distance

        if img.ndim == 2:
            # Bilinear interpolation for grayscale
            result = (img[r0][:, c0] * (1 - dr) * (1 - dc) +
                      img[r0][:, c1] * (1 - dr) *      dc  +
                      img[r1][:, c0] *      dr  * (1 - dc) +
                      img[r1][:, c1] *      dr  *      dc)
        else:
            # Apply per channel for color images
            result = np.stack([
                (img[r0][:, c0, ch] * (1 - dr) * (1 - dc) +
                 img[r0][:, c1, ch] * (1 - dr) *      dc  +
                 img[r1][:, c0, ch] *      dr  * (1 - dc) +
                 img[r1][:, c1, ch] *      dr  *      dc)
                for ch in range(img.shape[2])
            ], axis=-1)

        return result.astype(np.uint8)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'nearest' or 'bilinear'.")
