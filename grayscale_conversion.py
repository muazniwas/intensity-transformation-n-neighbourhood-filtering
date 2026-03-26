import cv2
import matplotlib.pyplot as plt

imgPath = 'images/woman_open_window.png'
img = cv2.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not load '{imgPath}'. Check the file path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/grayscale_output.png', dpi=150, bbox_inches='tight')
plt.show()

cv2.imwrite('outputs/grayscale.png', gray)
print("Done. Outputs saved: grayscale.png, grayscale_output.png")
