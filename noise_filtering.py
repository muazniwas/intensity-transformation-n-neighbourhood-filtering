import cv2
import matplotlib.pyplot as plt

imgPath = 'images/salt_n_pepper_emma.png'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

# Gaussian smoothing — kernel size 5x5, sigma=1
gaussian = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)

# Median filtering — kernel size 5x5
median = cv2.medianBlur(img, ksize=5)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Salt & Pepper Noise')
axes[0].axis('off')

axes[1].imshow(gaussian, cmap='gray')
axes[1].set_title('Gaussian Smoothing\n(5×5, σ=1)')
axes[1].axis('off')

axes[2].imshow(median, cmap='gray')
axes[2].set_title('Median Filtering\n(5×5)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/noise_filtering_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/gaussian_smoothed_emma.png', gaussian)
cv2.imwrite('outputs/median_filtered_emma.png', median)
print("Done. Outputs saved: gaussian_smoothed_emma.png, median_filtered_emma.png, noise_filtering_output.png")
