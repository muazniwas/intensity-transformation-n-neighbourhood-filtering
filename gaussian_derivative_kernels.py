import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gaussian_kernel import gaussian_kernel


def gaussian_derivative_kernels(size, sigma):
    """
    Compute normalized 5x5 derivative-of-Gaussian kernels in x and y.

    From the partial derivatives derived analytically:
        dG/dx = -(x / sigma^2) * G(x, y)
        dG/dy = -(y / sigma^2) * G(x, y)

    Each kernel is normalized by dividing by the sum of its absolute values
    so the net response on a uniform region is zero (derivative property).
    """
    half = size // 2
    ax = np.arange(-half, half + 1)
    x, y = np.meshgrid(ax, ax)

    G = gaussian_kernel(size, sigma)

    dG_dx = -(x / sigma**2) * G
    dG_dy = -(y / sigma**2) * G

    # Normalize by sum of absolute values
    dG_dx /= np.abs(dG_dx).sum()
    dG_dy /= np.abs(dG_dy).sum()

    return dG_dx, dG_dy


dG_dx, dG_dy = gaussian_derivative_kernels(size=5, sigma=2)

np.set_printoptions(precision=6, suppress=True)
print("5×5 Derivative-of-Gaussian Kernel — x-direction (dG/dx):\n")
print(dG_dx)

print("\n5×5 Derivative-of-Gaussian Kernel — y-direction (dG/dy):\n")
print(dG_dy)

print(f"\nSum of dG/dx values : {dG_dx.sum():.6f}  (should be ≈ 0)")
print(f"Sum of dG/dy values : {dG_dy.sum():.6f}  (should be ≈ 0)")

# --- 3D surface plot of 51×51 derivative-of-Gaussian (x-direction) ---
dG_dx_51, _ = gaussian_derivative_kernels(size=51, sigma=2)

half = 51 // 2
ax = np.arange(-half, half + 1)
x, y = np.meshgrid(ax, ax)

fig = plt.figure(figsize=(9, 6))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot_surface(x, y, dG_dx_51, cmap=cm.coolwarm, edgecolor='none')

ax3d.set_title('51×51 Derivative-of-Gaussian Kernel — x-direction (σ = 2)', fontsize=12)
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('Coefficient')

plt.tight_layout()
plt.savefig('outputs/gaussian_derivative_kernel_3d.png', dpi=150, bbox_inches='tight')
plt.show()
print("Output saved: gaussian_derivative_kernel_3d.png")
