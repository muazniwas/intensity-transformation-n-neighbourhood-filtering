import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_zoom import zoom


def normalized_ssd(img_a, img_b):
    """
    Normalized Sum of Squared Differences between two same-size images.
        SSD = sum((A - B)^2) / N
    where N is the total number of pixels.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    return np.sum((a - b) ** 2) / a.size


def evaluate(original_path, small_path):
    original_bgr = cv2.imread(original_path)
    small_bgr    = cv2.imread(small_path)

    if original_bgr is None:
        raise FileNotFoundError(f"Could not load '{original_path}'")
    if small_bgr is None:
        raise FileNotFoundError(f"Could not load '{small_path}'")

    # Convert BGR → RGB for display
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    small    = cv2.cvtColor(small_bgr,    cv2.COLOR_BGR2RGB)

    H, W = original.shape[:2]
    h, w = small.shape[:2]
    s = W / w   # scale factor to match original size

    zoomed_nn = zoom(small, s=s, method='nearest')
    zoomed_bl = zoom(small, s=s, method='bilinear')

    # Crop to exact original size in case of rounding differences
    zoomed_nn = zoomed_nn[:H, :W]
    zoomed_bl = zoomed_bl[:H, :W]

    ssd_nn = normalized_ssd(original, zoomed_nn)
    ssd_bl = normalized_ssd(original, zoomed_bl)

    return original, small, zoomed_nn, zoomed_bl, s, ssd_nn, ssd_bl


# Image pairs: (original, small)
pairs = [
    ('images/q7images/im01.png',    'images/q7images/im01small.png'),
    ('images/q7images/im02.png',    'images/q7images/im02small.png'),
    ('images/q7images/im03.png',    'images/q7images/im03small.png'),
    ('images/q7images/taylor.jpg',  'images/q7images/taylor_small.jpg'),
]

print(f"{'Image':<20} {'Scale':>6} {'SSD (Nearest)':>16} {'SSD (Bilinear)':>16}")
print("-" * 62)

for orig_path, small_path in pairs:
    original, small, zoomed_nn, zoomed_bl, s, ssd_nn, ssd_bl = evaluate(orig_path, small_path)

    name = orig_path.split('/')[-1]
    stem = name.rsplit('.', 1)[0]
    print(f"{name:<20} {s:>6.2f} {ssd_nn:>16.4f} {ssd_bl:>16.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original, cmap=None)
    axes[0].set_title(f'Original\n{original.shape[1]}×{original.shape[0]}')
    axes[0].axis('off')

    axes[1].imshow(small, cmap=None)
    axes[1].set_title(f'Small\n{small.shape[1]}×{small.shape[0]}')
    axes[1].axis('off')

    axes[2].imshow(zoomed_nn, cmap=None)
    axes[2].set_title(f'Nearest (s={s:.2f})\nSSD={ssd_nn:.2f}')
    axes[2].axis('off')

    axes[3].imshow(zoomed_bl, cmap=None)
    axes[3].set_title(f'Bilinear (s={s:.2f})\nSSD={ssd_bl:.2f}')
    axes[3].axis('off')

    plt.tight_layout()
    out_path = f'outputs/zoom_ssd_{stem}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out_path}")

print("\nDone.")
