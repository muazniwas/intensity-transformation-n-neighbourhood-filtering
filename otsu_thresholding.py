import cv2
import matplotlib.pyplot as plt

imgPath = 'images/woman_open_window.png'
img = cv2.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu's thresholding
# THRESH_BINARY_INV is used so the foreground (woman + room) is white (255)
# and the bright window background is black (0).
thresh_val, binary_mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

print(f"Otsu's threshold value: {thresh_val:.0f}")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

axes[2].imshow(binary_mask, cmap='gray')
axes[2].set_title(f"Otsu Binary Mask\n(threshold = {thresh_val:.0f})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/otsu_thresholding_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/otsu_binary_mask.png', binary_mask)
print("Outputs saved: otsu_binary_mask.png, otsu_thresholding_output.png")
