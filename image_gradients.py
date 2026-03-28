import cv2
import numpy as np
import matplotlib.pyplot as plt

from gaussian_derivative_kernels import gaussian_derivative_kernels


imgPath = 'images/einstein.png'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

# Compute 5x5 derivative-of-Gaussian kernels for sigma=2
dG_dx, dG_dy = gaussian_derivative_kernels(size=5, sigma=2)

# Convolve image with each kernel to get gradients
# ddepth=-1 keeps the same depth; use float32 to preserve negative values
Gx = cv2.filter2D(img.astype(np.float32), ddepth=-1, kernel=dG_dx)
Gy = cv2.filter2D(img.astype(np.float32), ddepth=-1, kernel=dG_dy)

# Gradient magnitude
G_mag = np.sqrt(Gx**2 + Gy**2)

# Normalize to [0, 255] for display
def normalize(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    return arr.astype(np.uint8)

Gx_display   = normalize(Gx)
Gy_display   = normalize(Gy)
Gmag_display = normalize(G_mag)

# --- Sobel gradients via OpenCV ---
Sx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
Sy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
S_mag = np.sqrt(Sx**2 + Sy**2)

Sx_display   = normalize(Sx)
Sy_display   = normalize(Sy)
Smag_display = normalize(S_mag)

# --- Plot: Derivative-of-Gaussian row ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(Gx_display, cmap='gray')
axes[0, 1].set_title('DoG — Gx (dG/dx)')
axes[0, 1].axis('off')

axes[0, 2].imshow(Gy_display, cmap='gray')
axes[0, 2].set_title('DoG — Gy (dG/dy)')
axes[0, 2].axis('off')

axes[0, 3].imshow(Gmag_display, cmap='gray')
axes[0, 3].set_title('DoG — Gradient Magnitude')
axes[0, 3].axis('off')

# --- Plot: Sobel row ---
axes[1, 0].imshow(img, cmap='gray')
axes[1, 0].set_title('Original')
axes[1, 0].axis('off')

axes[1, 1].imshow(Sx_display, cmap='gray')
axes[1, 1].set_title('Sobel — Sx')
axes[1, 1].axis('off')

axes[1, 2].imshow(Sy_display, cmap='gray')
axes[1, 2].set_title('Sobel — Sy')
axes[1, 2].axis('off')

axes[1, 3].imshow(Smag_display, cmap='gray')
axes[1, 3].set_title('Sobel — Gradient Magnitude')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('outputs/image_gradients_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/gradient_dog_x.png', Gx_display)
cv2.imwrite('outputs/gradient_dog_y.png', Gy_display)
cv2.imwrite('outputs/gradient_dog_magnitude.png', Gmag_display)
cv2.imwrite('outputs/gradient_sobel_x.png', Sx_display)
cv2.imwrite('outputs/gradient_sobel_y.png', Sy_display)
cv2.imwrite('outputs/gradient_sobel_magnitude.png', Smag_display)

print("Done. Outputs saved to outputs/.")
