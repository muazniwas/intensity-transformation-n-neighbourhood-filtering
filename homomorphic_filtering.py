import cv2
import numpy as np
import matplotlib.pyplot as plt


def homomorphic_filter(img, gamma_l=0.5, gamma_h=2.0, cutoff=30, c=1.0):
    """
    Homomorphic filter for illumination correction.

    Based on the image model: f(x,y) = i(x,y) * r(x,y)
        where i = illumination (low frequency), r = reflectance (high frequency)

    Algorithm:
        1. Log transform   : z(x,y) = ln(f(x,y))  →  separates i and r additively
        2. FFT             : Z(u,v) = FFT(z(x,y))
        3. Filter          : S(u,v) = H(u,v) * Z(u,v)
                             H attenuates low freqs (illumination, γL < 1)
                             H amplifies high freqs (reflectance, γH > 1)
        4. Inverse FFT     : s(x,y) = IFFT(S(u,v))
        5. Exp transform   : g(x,y) = exp(s(x,y))  →  returns to intensity domain

    The filter H(u,v) is a high-emphasis Gaussian filter:
        H(u,v) = (γH - γL) * [1 - exp(-c * D²(u,v) / D0²)] + γL

    Parameters
    ----------
    img     : grayscale uint8 image
    gamma_l : gain for low frequencies  (< 1 suppresses illumination variation)
    gamma_h : gain for high frequencies (> 1 enhances reflectance/edges)
    cutoff  : cutoff frequency D0 in pixels
    c       : sharpness of filter transition

    Returns
    -------
    Corrected grayscale image as uint8.
    """
    # Step 1 — Log transform (add 1 to avoid log(0))
    img_log = np.log1p(img.astype(np.float64))

    # Step 2 — DFT (shift zero frequency to centre)
    dft     = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    # Step 3 — Build high-emphasis homomorphic filter H(u,v)
    H, W = img.shape
    u = np.arange(-H // 2, H // 2) if H % 2 == 0 else np.arange(-(H // 2), H // 2 + 1)
    v = np.arange(-W // 2, W // 2) if W % 2 == 0 else np.arange(-(W // 2), W // 2 + 1)
    V, U = np.meshgrid(v, u)
    D_sq = U**2 + V**2

    H_filter = (gamma_h - gamma_l) * (1 - np.exp(-c * D_sq / cutoff**2)) + gamma_l

    # Apply filter in frequency domain
    filtered_shift = H_filter * dft_shift

    # Step 4 — Inverse DFT
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    filtered = np.real(filtered)

    # Step 5 — Exponential (inverse of log)
    result = np.expm1(filtered)

    # Normalize to [0, 255]
    result = result - result.min()
    result = result / result.max() * 255
    return result.astype(np.uint8)


# --- Apply to highlights_and_shadows image ---
imgPath = 'images/highlights_and_shadows.jpg'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

corrected = homomorphic_filter(img, gamma_l=0.5, gamma_h=2.0, cutoff=30, c=1.0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original (Non-uniform Illumination)')
axes[0].axis('off')

axes[1].imshow(corrected, cmap='gray')
axes[1].set_title('Homomorphic Filtered\n(γL=0.5, γH=2.0, D0=30)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/homomorphic_filter_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/homomorphic_filtered.png', corrected)
print("Done. Outputs saved: homomorphic_filtered.png, homomorphic_filter_output.png")
