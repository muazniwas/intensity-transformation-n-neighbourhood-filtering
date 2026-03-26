import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def gaussian_kernel(size, sigma):
    """
    Compute a normalized 2D Gaussian kernel.
        G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
    The kernel is normalized so all values sum to 1.
    """
    half = size // 2
    ax = np.arange(-half, half + 1)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


kernel = gaussian_kernel(size=5, sigma=2)

np.set_printoptions(precision=4, suppress=True)
print("5x5 Gaussian Kernel (σ = 2):\n")
print(kernel)
print(f"\nSum of all values: {kernel.sum():.6f}")

# --- 3D surface plot of 51x51 kernel ---
kernel_51 = gaussian_kernel(size=51, sigma=2)

half = 51 // 2
ax = np.arange(-half, half + 1)
x, y = np.meshgrid(ax, ax)

fig = plt.figure(figsize=(9, 6))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot_surface(x, y, kernel_51, cmap=cm.viridis, edgecolor='none')

ax3d.set_title('51×51 Gaussian Kernel (σ = 2)', fontsize=13)
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('Coefficient')

plt.tight_layout()
plt.savefig('outputs/gaussian_kernel_3d.png', dpi=150, bbox_inches='tight')
plt.show()
print("Output saved: gaussian_kernel_3d.png")
