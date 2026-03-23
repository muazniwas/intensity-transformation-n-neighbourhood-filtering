import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_correction(img, gamma):
    """Apply gamma correction: s = (r/255)^gamma * 255"""
    normalized = img / 255.0
    corrected = np.power(normalized, gamma)
    return np.uint8(corrected * 255)


# Load image in grayscale
imgPath = 'images/runway.png'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load image. Check the file path.")

# Apply gamma corrections
gamma_05 = gamma_correction(img, gamma=0.5)
gamma_2  = gamma_correction(img, gamma=2.0)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(gamma_05, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Gamma Correction (γ = 0.5)\n[Brightened]')
axes[1].axis('off')

axes[2].imshow(gamma_2, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Gamma Correction (γ = 2.0)\n[Darkened]')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/gamma_correction_output.png', dpi=150, bbox_inches='tight')
plt.show()

# Save individual output images
cv2.imwrite('outputs/gamma_0.5.jpg', gamma_05)
cv2.imwrite('outputs/gamma_2.0.jpg', gamma_2)

print("Done. Outputs saved: gamma_0.5.jpg, gamma_2.0.jpg, gamma_correction_output.png")
