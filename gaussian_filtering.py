import cv2
import numpy as np
import matplotlib.pyplot as plt

from gaussian_kernel import gaussian_kernel


imgPath = 'images/einstein.png'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

kernel = gaussian_kernel(size=51, sigma=2)

# Apply Gaussian filter via convolution
blurred = cv2.filter2D(img, ddepth=-1, kernel=kernel)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(blurred, cmap='gray')
axes[1].set_title('Gaussian Filtered (51×51, σ = 2)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/gaussian_filtering_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/gaussian_filtered_einstein.png', blurred)
print("Done. Outputs saved: gaussian_filtered_einstein.png, gaussian_filtering_output.png")
