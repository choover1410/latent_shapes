import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import skfmm
from burn import burn_grain
import random
from scipy.interpolate import interp1d


def generate_random_shape(num_sym=None, num_points=None, num_corners=None, amp_low=None, amp_high=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    # Generate piecewise random perturbations at fixed angles to create sharp edges
    corner_angles = np.linspace(0, 2*np.pi, num_corners*2*num_sym, endpoint=False)

    corner_values = np.random.uniform((amp_low)*100, (1+amp_high)*100, num_corners)
    max_corner = max(corner_values)
    double_values = np.append(corner_values, corner_values[::-1]) # now symmetric
    values = np.array([])
    for i in range(num_sym):
        values = np.append(values, double_values)

    # Interpolate smoothly between sharp edges
    r = np.interp(t, corner_angles, values, period=2*np.pi)

    #print(f"t: {t.shape}")
    #print(f"r: {r.shape}")

    # Convert to complex points
    x = r * np.cos(t)
    y = r * np.sin(t)
    shape_points = x + 1j * y  # Complex representation

    return shape_points, max_corner

def compute_fourier_series(points):
    N = len(points)
    fourier_coeffs = np.fft.fft(points) / N  # Normalize DFT
    frequencies = np.fft.fftfreq(N) * N # Convert to integer frequencies
    
    return fourier_coeffs, frequencies


def reconstruct_shape(fourier_coeffs, frequencies, num_points):
    t = np.linspace(0, 2 * np.pi, num_points)
    reconstructed = np.zeros(num_points, dtype=complex)

    for coef, freq in zip(fourier_coeffs, frequencies):
        reconstructed += coef * np.exp(1j * freq * t)

    return reconstructed


def fill_shape(edge_matrix):
    """
    Takes a 2D matrix where edge points are marked (1) and fills the interior.
    """
    filled_matrix = binary_fill_holes(edge_matrix).astype(int)
    return filled_matrix


def complex_to_matrix(complex_array, grid_size):
    """
    Converts an array of complex numbers into an NxN 2D matrix for visualization.
    """
    matrix = np.zeros((grid_size, grid_size), dtype=int)
    real_vals = np.real(complex_array)
    imag_vals = np.imag(complex_array)
    
    # Normalize to grid indices
    x_indices = ((real_vals - real_vals.min()) / (real_vals.max() - real_vals.min()) * (grid_size - 1)).astype(int)
    y_indices = ((imag_vals - imag_vals.min()) / (imag_vals.max() - imag_vals.min()) * (grid_size - 1)).astype(int)
    
    matrix[y_indices, x_indices] = 1  # Mark points in the matrix
    return matrix

def resample_timeseries(data, num_points):
    if len(data) == 0 or num_points <= 0:
        raise ValueError("Input data must be non-empty and num_points must be positive.")

    x_original = np.linspace(0, 1, len(data))
    x_resampled = np.linspace(0, 1, num_points)
    resampled_data = np.interp(x_resampled, x_original, data)

    return resampled_data


def generate_burnback_curve_image(y_points, width=128, height=128):
    # Generate linearly spaced X values from 0 to 1
    x = np.linspace(0, 1, len(y_points))
    y = np.array(y_points)

    # Interpolate to create a smooth curve
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    x_new = np.linspace(0, 1, width)
    y_new = f(x_new)

    # Scale y_new to fit the image height without stretching
    y_new = y_new * height
    y_new = np.clip(y_new, 0, height - 1)

    # Create an empty image
    image = np.ones((height, width), dtype=np.uint8) * 255

    # Fill below the curve with black
    for i in range(width):
        y_coord = int(height - y_new[i])  # Flip y-axis for image coordinates
        image[y_coord:, i] = 0

    # Save the image
    return image


def main():
    W = 600 # X Image size
    N=100 # number of burn contours

    NUM_POINTS = 8000
    AMP_LOW = random.randint(20, 40) / 100
    AMP_HIGH = random.randint(60, 75) / 100
    NUM_CORNERS = random.randint(2, 4)
    NUM_SYM = random.randint(3, 5)

    polygon_points, max_corner_val = generate_random_shape(num_sym=NUM_SYM, num_points=NUM_POINTS, num_corners=NUM_CORNERS,  amp_low=AMP_LOW, amp_high=AMP_HIGH, seed=None)
    
    fourier_coeffs, frequencies = compute_fourier_series(polygon_points)
    reconstructed_points = reconstruct_shape(fourier_coeffs, frequencies, NUM_POINTS)

    real_pts = polygon_points.real
    imag_pts = polygon_points.imag


    # Plotting for debugging
    if False:
        CUTOFF = int(NUM_POINTS / 2)
        # Plot original and reconstructed shape
        plt.figure(figsize=(8, 8))
        plt.plot(polygon_points.real, polygon_points.imag, 'rx', label="Original Shape", alpha=0.6)
        plt.plot(reconstructed_points.real, reconstructed_points.imag, label="Reconstructed Shape", linewidth=2)
        plt.scatter(fourier_coeffs.real, fourier_coeffs.imag, color="red", label="Fourier Coefficients", s=30)
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.show()

        reals = fourier_coeffs[:CUTOFF].real
        imags = fourier_coeffs[:CUTOFF].imag

        # Plot components (real, imag)


        plt.figure()
        magnitudes = []
        for i in range(len(reals)):
            mag = np.sqrt(reals[i]**2 + imags[i]**2)
            mag = mag / 200 # normalize by max radius
            if mag < 1e-6:
                mag = 1e-6
            magnitudes.append(mag)
        plt.plot(frequencies[:CUTOFF][:500], magnitudes[:500])
        plt.yscale("log")
        plt.title('DFT Mag')
        plt.grid()

        # Plot phase
        plt.figure()
        phases = []
        for i in range(len(reals)):
            phase = np.arctan2(imags[i], reals[i])
            phases.append(phase)
        plt.plot(frequencies[:CUTOFF][:500], phases[:500])
        plt.title('DFT Phase')
        plt.grid()
        
        plt.show()


    # Everything below here is for the burnback curve
    shape_size = random.randint(int(0.25*W), int(0.9*W))

    outline = complex_to_matrix(reconstructed_points, shape_size)
    filled_matrix = fill_shape(outline)

    shift_factor = (W - shape_size)//2
    X, Y = np.meshgrid(np.linspace(-1.0,1.0,W), np.linspace(-1.0,1.0,W))

    phi = 1 * np.ones_like(X)
    for row in range(len(filled_matrix)):
        for col in range(len(filled_matrix)):
            if filled_matrix[row, col] == 1:
                phi[row + shift_factor, col + shift_factor] = 0

    integral_y, phi = burn_grain(X,Y,phi,N,1e-2)
    resampled = resample_timeseries(integral_y, 50)


    return phi, resampled, real_pts, imag_pts, shape_size, max_corner_val

if __name__ == "__main__":
    main()