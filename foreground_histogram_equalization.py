import cv2
import numpy as np
import matplotlib.pyplot as plt

imgPath = 'images/woman_open_window.png'
img = cv2.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 1: Get foreground mask via Otsu thresholding ---
thresh_val, fg_mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
print(f"Otsu threshold: {thresh_val:.0f}")

# --- Step 2: Histogram equalization on foreground pixels only ---
# Build CDF from foreground pixels only, then apply the mapping back.
fg_pixels = gray[fg_mask == 255]

hist, bins = np.histogram(fg_pixels, bins=256, range=(0, 256))
cdf = hist.cumsum()
cdf_min = cdf[cdf > 0].min()
n_fg = fg_pixels.size

# Equalization formula: round((cdf(r) - cdf_min) / (n_fg - cdf_min) * 255)
lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    if cdf[i] > 0:
        lut[i] = round((cdf[i] - cdf_min) / (n_fg - cdf_min) * 255)

# Apply LUT only to foreground pixels; background stays unchanged
result = gray.copy()
result[fg_mask == 255] = lut[gray[fg_mask == 255]]

# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('Original (Grayscale)')
axes[0, 0].axis('off')

axes[0, 1].imshow(fg_mask, cmap='gray')
axes[0, 1].set_title(f'Foreground Mask\n(Otsu threshold = {thresh_val:.0f})')
axes[0, 1].axis('off')

axes[0, 2].imshow(result, cmap='gray')
axes[0, 2].set_title('Foreground Histogram Equalized')
axes[0, 2].axis('off')

axes[1, 0].hist(gray.ravel(), bins=256, range=(0, 255), color='steelblue', edgecolor='none')
axes[1, 0].set_title('Histogram — Original (full image)')
axes[1, 0].set_xlabel('Pixel Intensity')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(fg_pixels, bins=256, range=(0, 255), color='seagreen', edgecolor='none')
axes[1, 1].set_title('Histogram — Foreground pixels only')
axes[1, 1].set_xlabel('Pixel Intensity')
axes[1, 1].set_ylabel('Frequency')

axes[1, 2].hist(result[fg_mask == 255], bins=256, range=(0, 255), color='darkorange', edgecolor='none')
axes[1, 2].set_title('Histogram — Equalized foreground')
axes[1, 2].set_xlabel('Pixel Intensity')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/foreground_histogram_equalization_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/foreground_histogram_equalization.png', result)

print("\nOutputs saved: foreground_histogram_equalization.png, foreground_histogram_equalization_output.png")
