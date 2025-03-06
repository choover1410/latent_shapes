import numpy as np
import matplotlib.pyplot as plt


def generate_random_shape(num_points=300, num_corners=10, amplitude=0.75, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Parameter t evenly spaced around the unit circle
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Generate piecewise random perturbations at fixed angles to create sharp edges
    corner_angles = np.linspace(0, 2 * np.pi, num_corners, endpoint=False)
    corner_values = np.random.uniform(1 - amplitude, 1 + amplitude, num_corners)

    # Interpolate smoothly between sharp edges
    r = np.interp(t, corner_angles, corner_values, period=2 * np.pi)

    # Convert to complex points
    x = r * np.cos(t)
    y = r * np.sin(t)
    shape_points = x + 1j * y  # Complex representation

    return shape_points

def compute_fourier_series(points, num_coeffs):
    N = len(points)
    fourier_coeffs = np.fft.fft(points) / N  # Normalize DFT
    frequencies = np.fft.fftfreq(N) * N  # Convert to integer frequencies
    #indices = np.argsort(np.abs(frequencies))[:num_coeffs]  # Take the most relevant coefficients
    
    #return fourier_coeffs[indices], frequencies[indices]
    return fourier_coeffs, frequencies


def reconstruct_shape(fourier_coeffs, frequencies, num_points):
    t = np.linspace(0, 2 * np.pi, num_points)
    reconstructed = np.zeros(num_points, dtype=complex)

    for coef, freq in zip(fourier_coeffs, frequencies):
        reconstructed += coef * np.exp(1j * freq * t)

    return reconstructed

AMP = 0.8
NUM_CORNERS = 10
NUM_POINTS = 500
NUM_COEFFS = 200

polygon_points = generate_random_shape(num_points=NUM_POINTS, num_corners=NUM_CORNERS, amplitude=AMP, seed=None)
fourier_coeffs, frequencies = compute_fourier_series(polygon_points, NUM_COEFFS)
reconstructed_points = reconstruct_shape(fourier_coeffs, frequencies, NUM_POINTS)

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
reals = fourier_coeffs.real
for i,_ in enumerate(reals):
    if reals[i] < 1e-10:
        reals[i] = 0
plt.scatter(frequencies, reals)
print(f"Freqs: {len(frequencies)}")
print(f"Real: {len(fourier_coeffs.real)}")
plt.yscale("log")
plt.title('Real Coeffs.')
plt.grid()

plt.figure()
imags = fourier_coeffs.imag
for i,_ in enumerate(imags):
    if imags[i] < 1e-10:
        imags[i] = 0
plt.scatter(frequencies, imags)
print(f"Freqs: {len(frequencies)}")
print(f"Imag: {len(fourier_coeffs.imag)}")
plt.yscale("log")
plt.title('Imag. Coeffs.')
plt.grid()
plt.show()
