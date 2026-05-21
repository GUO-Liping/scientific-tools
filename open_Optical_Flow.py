import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== Optical Flow Parameters ======================
pixels_per_meter = 881 / 0.6        # Calibration factor (px/m)
frame_rate = 120.0                  # Frame rate
clahe_clip = 3.0                     # CLAHE clip limit
clahe_tile = (8, 8)                  # CLAHE grid size
gaussian_ksize = (3, 3)             # Gaussian blur kernel size

farneback_params = {
    'pyr_scale': 0.5,
    'levels': 5,
    'winsize': 15,
    'iterations': 4,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}

visual_step = 12  # Subsample factor for vector field

# ====================== Functions ======================

def enhance_frame(frame_gray):
    """Enhance image using CLAHE and Gaussian blur."""
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    frame_enh = clahe.apply(frame_gray)
    frame_enh = cv2.GaussianBlur(frame_enh, gaussian_ksize, 0)
    return frame_enh

def compute_optical_flow(frame_a, frame_b):
    """Compute Farneback optical flow; return horizontal, vertical velocities and magnitude."""
    flow = cv2.calcOpticalFlowFarneback(frame_a, frame_b, None, **farneback_params)
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)
    return u, v, magnitude

def convert_to_physical(u, v, dt):
    """Convert pixel/frame velocities to m/s."""
    u_phys = u / dt / pixels_per_meter
    v_phys = v / dt / pixels_per_meter
    magnitude_phys = np.sqrt(u_phys**2 + v_phys**2)
    return u_phys, v_phys, magnitude_phys

def visualize(frame_gray, frame_enh, u, v, magnitude_phys):
    """Visualize original, enhanced frames, velocity magnitude, and vector field."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    plt.tight_layout(pad=3.5)

    # Original frame
    axes[0, 0].imshow(frame_gray, cmap='gray', origin='upper')
    axes[0, 0].set_title("Original Frame")

    # Enhanced frame
    axes[0, 1].imshow(frame_enh, cmap='gray', origin='upper')
    axes[0, 1].set_title("Enhanced Frame (CLAHE + GaussianBlur)")

    # Velocity magnitude
    im = axes[1, 0].imshow(magnitude_phys, cmap='jet', origin='upper')
    axes[1, 0].set_title("Velocity Magnitude (m/s)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Vector field (subsampled)
    y_idx, x_idx = np.mgrid[0:u.shape[0]:visual_step, 0:u.shape[1]:visual_step]
    axes[1, 1].imshow(frame_gray, cmap='gray', origin='upper')
    axes[1, 1].quiver(x_idx, y_idx, u[y_idx, x_idx], -v[y_idx, x_idx],  # flip v for plotting
                      color='red', scale=600, width=0.005)
    axes[1, 1].set_title("Velocity Vector Field")

    plt.show()

def save_results(frame_start, frame_end, u_phys, v_phys, magnitude_phys):
    """Save optical flow results to npz file."""
    filename = f"optical_flow_{frame_start}_{frame_end}.npz"
    np.savez(filename,
             u_phys=u_phys, v_phys=v_phys, magnitude_phys=magnitude_phys,
             params={
                 'frame_rate': frame_rate,
                 'pixels_per_meter': pixels_per_meter,
                 'clahe_clip': clahe_clip,
                 'clahe_tile': clahe_tile,
                 'gaussian_ksize': gaussian_ksize,
                 'farneback_params': farneback_params
             })
    print(f"Results saved to {filename}")

# ====================== Main Script ======================

if __name__ == "__main__":
    frame_start = 4335
    frame_end = 4336
    dt = (frame_end - frame_start) / frame_rate
    print(f"Processing frames {frame_start} -> {frame_end}, dt = {dt:.4f} s")

    # Read frames from images
    frame_a_color = cv2.imread(f'frame_{frame_start}_crop.png')
    frame_b_color = cv2.imread(f'frame_{frame_end}_crop.png')

    # Ensure same size
    if frame_a_color.shape != frame_b_color.shape:
        frame_b_color = cv2.resize(frame_b_color, (frame_a_color.shape[1], frame_a_color.shape[0]))

    # Convert to grayscale
    frame_a_gray = cv2.cvtColor(frame_a_color, cv2.COLOR_BGR2GRAY)
    frame_b_gray = cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2GRAY)

    # Enhance images
    frame_a_enh = enhance_frame(frame_a_gray)
    frame_b_enh = enhance_frame(frame_b_gray)
    print("Image enhancement complete.")

    # Compute optical flow
    u, v, magnitude = compute_optical_flow(frame_a_enh, frame_b_enh)

    # Convert to physical units
    u_phys, v_phys, magnitude_phys = convert_to_physical(u, v, dt)
    print(f"Maximum velocity: {magnitude_phys.max():.3f} m/s")

    # Visualize (flip v for vector field)
    visualize(frame_b_gray, frame_b_enh, u, v, magnitude_phys)

    # Save results
    save_results(frame_start, frame_end, u_phys, v_phys, magnitude_phys)