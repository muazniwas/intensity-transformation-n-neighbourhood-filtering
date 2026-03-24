import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gamma value — γ < 1 brightens shadows and reveals detail in dark regions.
# γ = 0.5 is chosen for 'highlights_and_shadows.jpg' to lift the dark areas
# while preserving color information by operating only on the L* channel.
GAMMA = 0.5


def gamma_correction_L(img_bgr, gamma):
    """
    Apply gamma correction to the L* channel of the L*a*b* color space.
        L_out = (L_in / 255)^gamma * 255
    Color (a*, b*) channels are left unchanged.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    L, a, b = cv2.split(lab)
    L_corrected = np.power(L / 255.0, gamma) * 255.0

    lab_corrected = cv2.merge([L_corrected.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


# Load image
imgPath = 'images/highlights_and_shadows.jpg'
img = cv2.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

# Apply gamma correction on L* channel
result = gamma_correction_L(img, gamma=GAMMA)

# Convert BGR → RGB for matplotlib display
img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Extract L* channels for histogram
L_original  = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))[0]
L_corrected = cv2.split(cv2.cvtColor(result, cv2.COLOR_BGR2LAB))[0]

# Plot images and histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(result_rgb)
axes[0, 1].set_title(f'Gamma Correction on L* Channel (γ = {GAMMA})')
axes[0, 1].axis('off')

axes[1, 0].hist(L_original.ravel(), bins=256, range=(0, 255), color='steelblue', edgecolor='none')
axes[1, 0].set_title('Histogram — Original (L* channel)')
axes[1, 0].set_xlabel('Pixel Intensity')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(L_corrected.ravel(), bins=256, range=(0, 255), color='darkorange', edgecolor='none')
axes[1, 1].set_title(f'Histogram — Corrected L* channel (γ = {GAMMA})')
axes[1, 1].set_xlabel('Pixel Intensity')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/gamma_correction_lab_output.png', dpi=150, bbox_inches='tight')
plt.show()

# Save output
cv2.imwrite('outputs/gamma_correction_lab.jpg', result)

print(f"Done. γ = {GAMMA} applied to L* channel.")
print("Outputs saved: gamma_correction_lab.jpg, gamma_correction_lab_output.png")
