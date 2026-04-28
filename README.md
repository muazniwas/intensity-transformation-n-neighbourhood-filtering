# Intensity Transformation & Neighbourhood Filtering
IT5437 Computer Vision — Assignment 1

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy matplotlib
```

## Scripts

| File | Description | Image Used |
|------|-------------|------------|
| `gamma_correction.py` | Gamma correction with γ=0.5 and γ=2.0 | `runway.png` |
| `gamma_correction_lab.py` | Gamma correction on L* channel in L*a*b* space | `highlights_and_shadows.jpg` |
| `contrast_stretching.py` | Piecewise linear contrast stretching (r1=0.2, r2=0.8) | `runway.png` |
| `grayscale_conversion.py` | Convert colour image to grayscale | `woman_open_window.jpg` |
| `histogram_equalization.py` | Manual histogram equalization via CDF | `runway.png` |
| `foreground_histogram_equalization.py` | Histogram equalization restricted to foreground region | `woman_open_window.jpg` |
| `otsu_thresholding.py` | Otsu's method to obtain foreground binary mask | `woman_open_window.jpg` |
| `gaussian_kernel.py` | Compute and visualize a normalized Gaussian kernel | — |
| `gaussian_filtering.py` | Gaussian filtering — manual kernel vs cv2.GaussianBlur | `einstein.png` |
| `gaussian_derivative_kernels.py` | Derivative-of-Gaussian kernels + 3D surface plot | — |
| `image_gradients.py` | Image gradients via DoG kernels and cv2.Sobel | `einstein.png` |
| `image_zoom.py` | Zoom function — nearest-neighbor and bilinear interpolation | — |
| `zoom_ssd_test.py` | SSD evaluation of zoom methods against original images | `q7images/` |
| `noise_filtering.py` | Gaussian smoothing vs median filtering on noisy image | `salt_n_pepper_emma.png` |
| `image_sharpening.py` | Unsharp masking and Laplacian sharpening | `daisy.jpg` |
| `bilateral_filter.py` | Manual bilateral filter implementation | — |
| `bilateral_filter_test.py` | Gaussian vs cv2 bilateral vs manual bilateral | `jeniffer.jpg` |
| `homomorphic_filtering.py` | Homomorphic filter for illumination correction | `highlights_and_shadows.jpg` |

## Output
All output images are saved to the `outputs/` directory.
