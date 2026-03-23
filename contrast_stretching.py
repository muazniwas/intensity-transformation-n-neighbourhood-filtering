import cv2
import numpy as np
import matplotlib.pyplot as plt


def contrast_stretching(img, r1=0.2, r2=0.8):
    """
    Piecewise linear contrast stretching (normalized input/output):
        s(r) = 0                      for r < r1
        s(r) = (r - r1) / (r2 - r1)  for r1 <= r <= r2
        s(r) = 1                      for r > r2
    """
    r = img / 255.0
    s = np.where(r < r1, 0.0,
        np.where(r <= r2, (r - r1) / (r2 - r1), 1.0))
    return np.uint8(s * 255)


# Load image in grayscale
imgPath = 'images/runway.png'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load image. Check the file path.")

# Apply contrast stretching
contrast = contrast_stretching(img, r1=0.2, r2=0.8)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(contrast, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Contrast Stretching\n(r1=0.2, r2=0.8)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/contrast_stretching_output.png', dpi=150, bbox_inches='tight')
plt.show()

# Save output image
cv2.imwrite('outputs/contrast_stretching.jpg', contrast)

print("Done. Outputs saved: contrast_stretching.jpg, contrast_stretching_output.png")
