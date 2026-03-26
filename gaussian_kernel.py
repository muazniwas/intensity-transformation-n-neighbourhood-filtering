import numpy as np

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
