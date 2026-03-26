import cv2
import numpy as np
import matplotlib.pyplot as plt

from gaussian_kernel import gaussian_kernel


imgPath = 'images/einstein.png'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

kernel = gaussian_kernel(size=51, sigma=2)

# Apply Gaussian filter via manual convolution
blurred_manual = cv2.filter2D(img, ddepth=-1, kernel=kernel)

# Apply Gaussian filter via OpenCV's built-in GaussianBlur
blurred_cv = cv2.GaussianBlur(img, ksize=(51, 51), sigmaX=2, sigmaY=2)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(blurred_manual, cmap='gray')
axes[1].set_title('Manual Gaussian Kernel\n(51×51, σ = 2)')
axes[1].axis('off')

axes[2].imshow(blurred_cv, cmap='gray')
axes[2].set_title('cv2.GaussianBlur\n(51×51, σ = 2)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/gaussian_filtering_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/gaussian_filtered_manual.png', blurred_manual)
cv2.imwrite('outputs/gaussian_filtered_cv.png', blurred_cv)
print("Done. Outputs saved: gaussian_filtered_manual.png, gaussian_filtered_cv.png, gaussian_filtering_output.png")
