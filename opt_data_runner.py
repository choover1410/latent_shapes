import matplotlib.pyplot as plt
import numpy as np
import os
from fourier3 import main as fourier3
from datetime import datetime

from optimization.utils.parameterization import fit_catmullrom, get_catmullrom_points


for i in range(10000):
    print(f"Burning object {i} of {10000}")
    now = str(datetime.now().timestamp())
    now = now.replace('.', '')

    phi, y_timeseries, real_pts, imag_pts, shape_size, max_corner_val = fourier3()

    # TODO this aint quite right
    # after fixing, need to then scale by ([shape_size] / 1250)
    real_pts = real_pts / max_corner_val * (shape_size / 1250)
    imag_pts = imag_pts / max_corner_val * (shape_size / 1250)


    zipped_points = zip(real_pts, imag_pts)
    array_points = np.array(list(zipped_points))

    P_tensor = fit_catmullrom(array_points, 150)

    # For plotting/debug only
    X_fit = get_catmullrom_points(P_tensor.detach().reshape(-1, 2), num_sample_pts = 201).detach().numpy()

    #!!!!!!!!!!!!!!!
    testing = False
    train = True
    #!!!!!!!!!!!!!!!

    if train is True:
        data_file_path = 'optimization/data/train_data.npz'
    else:
        data_file_path = 'optimization/data/validate_data.npz'


    if testing is True:
        num_points = 100
        radius = 600/2
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        x_points = radius + radius * np.cos(angles)
        y_points = radius + radius * np.sin(angles)

        plt.imshow(phi, cmap="gray")
        plt.scatter(x_points, y_points, color="red", s=1)
        plt.title("Shape")
        plt.axis("equal")
        plt.show()

        plt.figure()
        plt.plot(X_fit[:, 0], X_fit[:, 1])
        plt.scatter(P_tensor[:, 0], P_tensor[:, 1])
        plt.title(f"Points (normalized) from Catmull-Roll Spine")
        plt.axis("equal")
        plt.grid()
        plt.show()

        plt.figure()
        plt.title("Burnback")
        plt.plot(y_timeseries)
        plt.grid()
        plt.show()

    if testing is False:
        X_all = []
        Y_all = []

        try:
            existing_data = np.load(data_file_path)
            file_size_bytes = os.path.getsize(data_file_path)
            print(f"Existing file size: {file_size_bytes/1024} KB")

            existing_X = existing_data['X']
            X_all.append(existing_X)
            existing_Y = existing_data['Y']
            Y_all.append(existing_Y)

        except FileNotFoundError as e:
            print(f"File not found, creating new file: {e}.")
            existing_X = np.empty((0, 150, 2))
            existing_Y = np.empty((0, 50))

        # Append new data
        X_all.append(P_tensor.numpy().reshape(1, 150, 2))   # Reshape to (1, 150, 2)
        Y_all.append(y_timeseries.reshape(1, 50))           # Reshape to (1, 50)

        X_all = np.vstack(X_all)
        Y_all = np.vstack(Y_all)

        # Store in a single data file
        np.savez(
            data_file_path,
            X=X_all,
            Y=Y_all
        )
