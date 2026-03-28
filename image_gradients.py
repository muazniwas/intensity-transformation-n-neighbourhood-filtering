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

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(Gx_display, cmap='gray')
axes[1].set_title('Horizontal Gradient (Gx)\ndG/dx')
axes[1].axis('off')

axes[2].imshow(Gy_display, cmap='gray')
axes[2].set_title('Vertical Gradient (Gy)\ndG/dy')
axes[2].axis('off')

axes[3].imshow(Gmag_display, cmap='gray')
axes[3].set_title('Gradient Magnitude\n√(Gx² + Gy²)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('outputs/image_gradients_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/gradient_x.png', Gx_display)
cv2.imwrite('outputs/gradient_y.png', Gy_display)
cv2.imwrite('outputs/gradient_magnitude.png', Gmag_display)
print("Done. Outputs saved: gradient_x.png, gradient_y.png, gradient_magnitude.png, image_gradients_output.png")
