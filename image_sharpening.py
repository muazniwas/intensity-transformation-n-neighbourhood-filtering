import cv2
import numpy as np
import matplotlib.pyplot as plt

imgPath = 'images/daisy.jpg'
img_bgr = cv2.imread(imgPath)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Unsharp masking (applied per channel) ---
# Sharpened = Original + k * (Original - Blurred)
# where (Original - Blurred) is the high-frequency "detail" layer
blurred  = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=2, sigmaY=2)
detail   = img.astype(np.float32) - blurred.astype(np.float32)

k = 1.5   # sharpening strength — increase for stronger effect
sharpened = np.clip(img.astype(np.float32) + k * detail, 0, 255).astype(np.uint8)

# --- Laplacian sharpening (applied per channel) ---
# Uses the Laplacian (second-order derivative) to detect and enhance edges
laplacian = np.stack([
    cv2.Laplacian(img[:, :, ch], ddepth=cv2.CV_32F, ksize=3)
    for ch in range(3)
], axis=-1)
sharpened_lap = np.clip(img.astype(np.float32) - laplacian, 0, 255).astype(np.uint8)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(np.clip(detail + 128, 0, 255).astype(np.uint8))
axes[1].set_title('Detail Layer\n(Original − Blurred)')
axes[1].axis('off')

axes[2].imshow(sharpened)
axes[2].set_title(f'Unsharp Masking (k={k})\n(5×5 Gaussian, σ=2)')
axes[2].axis('off')

axes[3].imshow(sharpened_lap)
axes[3].set_title('Laplacian Sharpening\n(3×3 kernel)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('outputs/image_sharpening_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/sharpened_unsharp.png', sharpened)
cv2.imwrite('outputs/sharpened_laplacian.png', sharpened_lap)
print("Done. Outputs saved: sharpened_unsharp.png, sharpened_laplacian.png, image_sharpening_output.png")
