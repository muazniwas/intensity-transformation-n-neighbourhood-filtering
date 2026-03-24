import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalize_histogram(img):
    """
    Equalize the histogram of a grayscale image.
    For color images, converts to L*a*b* and equalizes only the L* channel
    to avoid hue/saturation distortion.
    """
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_eq = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L_eq, a, b]), cv2.COLOR_LAB2BGR)


# Load image
imgPath = 'images/runway.png'
img = cv2.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

# Apply histogram equalization
result = equalize_histogram(img)

# Extract L* channels for histogram
L_original  = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))[0]
L_equalized = cv2.split(cv2.cvtColor(result, cv2.COLOR_BGR2LAB))[0]

# Convert BGR → RGB for display
img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Plot images and histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(result_rgb)
axes[0, 1].set_title('Histogram Equalized')
axes[0, 1].axis('off')

axes[1, 0].hist(L_original.ravel(), bins=256, range=(0, 255), color='steelblue', edgecolor='none')
axes[1, 0].set_title('Histogram — Original (L* channel)')
axes[1, 0].set_xlabel('Pixel Intensity')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(L_equalized.ravel(), bins=256, range=(0, 255), color='darkorange', edgecolor='none')
axes[1, 1].set_title('Histogram — Equalized (L* channel)')
axes[1, 1].set_xlabel('Pixel Intensity')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/histogram_equalization_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/histogram_equalization.jpg', result)
print("Done. Outputs saved: histogram_equalization.jpg, histogram_equalization_output.png")
