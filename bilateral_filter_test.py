import cv2
import matplotlib.pyplot as plt

from bilateral_filter import bilateral_filter

imgPath = 'images/jeniffer.jpg'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

# Gaussian smoothing
gaussian = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=75, sigmaY=75)

# OpenCV bilateral filter
bilateral_cv = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Manual bilateral filter
print("Applying manual bilateral filter (this may take a moment)...")
bilateral_manual = bilateral_filter(img, diameter=9, sigma_s=75, sigma_r=75)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(gaussian, cmap='gray')
axes[1].set_title('Gaussian Smoothing\n(9×9, σ=75)')
axes[1].axis('off')

axes[2].imshow(bilateral_cv, cmap='gray')
axes[2].set_title('cv2.bilateralFilter\n(d=9, σs=75, σr=75)')
axes[2].axis('off')

axes[3].imshow(bilateral_manual, cmap='gray')
axes[3].set_title('Manual Bilateral Filter\n(d=9, σs=75, σr=75)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('outputs/bilateral_filter_test_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/jeniffer_gaussian.png', gaussian)
cv2.imwrite('outputs/jeniffer_bilateral_cv.png', bilateral_cv)
cv2.imwrite('outputs/jeniffer_bilateral_manual.png', bilateral_manual)
print("Done. Outputs saved: jeniffer_gaussian.png, jeniffer_bilateral_cv.png, jeniffer_bilateral_manual.png")
