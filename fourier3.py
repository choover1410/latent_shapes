import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import skfmm
from burn import burn_grain
import random

def generate_random_shape(num_sym=None, num_points=None, num_corners=None, amp_low=None, amp_high=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Parameter t evenly spaced around the unit circle
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    # Generate piecewise random perturbations at fixed angles to create sharp edges
    corner_angles = np.linspace(0, 2*np.pi, num_corners*2*num_sym, endpoint=False)

    corner_values = np.random.uniform((1-amp_low)*100, (1+amp_high)*100, num_corners)
    double_values = np.append(corner_values, corner_values[::-1]) # now symmetric
    values = np.array([])
    for i in range(num_sym):
        values = np.append(values, double_values)

    print(f"len values: {len(values)}")
    print(f"len angles: {len(corner_angles)}")

    # Interpolate smoothly between sharp edges
    r = np.interp(t, corner_angles, values, period=2*np.pi)

    # Convert to complex points
    x = r * np.cos(t)
    y = r * np.sin(t)
    shape_points = x + 1j * y  # Complex representation

    return shape_points

def compute_fourier_series(points):
    N = len(points)
    fourier_coeffs = np.fft.fft(points) / N  # Normalize DFT
    frequencies = np.fft.fftfreq(N) * N  # Convert to integer frequencies
    
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

def main():
    NUM_POINTS = 10000
    AMP_LOW = random.randint(5, 99) / 100
    AMP_HIGH = random.randint(5, 99) / 100
    NUM_CORNERS = random.randint(2, 6)
    NUM_SYM = random.randint(1, 5)

    polygon_points = generate_random_shape(num_sym=NUM_SYM, num_points=NUM_POINTS, num_corners=NUM_CORNERS,  amp_low=AMP_LOW, amp_high=AMP_HIGH, seed=None)
    fourier_coeffs, frequencies = compute_fourier_series(polygon_points)
    reconstructed_points = reconstruct_shape(fourier_coeffs, frequencies, NUM_POINTS)

    """
    CUTOFF = int(NUM_POINTS / 2)
    # Plot original and reconstructed shape
    plt.figure(figsize=(8, 8))
    plt.plot(polygon_points.real, polygon_points.imag, 'rx', label="Original Shape", alpha=0.6)
    plt.plot(reconstructed_points.real, reconstructed_points.imag, label="Reconstructed Shape", linewidth=2)
    plt.scatter(fourier_coeffs.real, fourier_coeffs.imag, color="red", label="Fourier Coefficients", s=30)
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("Fourier Series Representation of a Smooth Closed-Loop Polygon")
    plt.show()

    plt.figure()
    reals = fourier_coeffs[:CUTOFF].real
    for i,_ in enumerate(reals):
        if reals[i] < 1e-7:
            reals[i] = 1e-7
    plt.plot(frequencies[:CUTOFF], reals)
    plt.yscale("log")
    plt.title('Real Coeffs.')
    plt.grid()

    plt.figure()
    imags = fourier_coeffs[:CUTOFF].imag
    for i,_ in enumerate(imags):
        if imags[i] < 1e-7:
            imags[i] = 1e-7
    plt.plot(frequencies[:CUTOFF], imags)
    plt.yscale("log")
    plt.title('Imag. Coeffs.')
    plt.grid()
    plt.show()
    """

    outline = complex_to_matrix(reconstructed_points, 500)
    filled_matrix = fill_shape(outline)


    N=50
    W = random.randint(750, 2000)
    shift_factor = (W - 500)//2
    X, Y = np.meshgrid(np.linspace(-1.0,1.0,W), np.linspace(-1.0,1.0,W))

    # Create initial geometry
    phi = 1 * np.ones_like(X)
    for row in range(len(filled_matrix)):
        for col in range(len(filled_matrix)):
            if filled_matrix[row, col] == 1:
                phi[row + shift_factor, col + shift_factor] = 0

    integral_y, phi = burn_grain(X,Y,phi,N,1)

    # THE END
    return phi, integral_y

if __name__ == "__main__":
    main()