import numpy as np
import torch

from .progress_bar import print_progress_bar


def fit_catmullrom(X, num_control_pts: int):
    X = X.reshape(-1, 2)
    X = torch.from_numpy(X)

    # Initialize control points for the spline
    idx = torch.linspace(0, X.shape[0] - 1, num_control_pts).to(torch.int)
    P = X[idx, :]

    # Make a list of the control points and set requires grad to True except first and last
    CP = P[1: num_control_pts - 1, :]
    CP.requires_grad = True

    # Set number of sample points to use for curve fitting
    num_sample_pts = 501

    # Setup the optimization problem
    learning_rate = 1e-3
    optimizer = torch.optim.Adam([CP], lr = learning_rate)
    loss_fn = curve_fit_loss

    # Training
    epochs = 100
    for epoch in range(1, epochs + 1):
        # Get the spline sample points
        P_tensor = torch.vstack([P[0, :], CP, P[-1, :]])
        X_fit = get_catmullrom_points(P_tensor, num_sample_pts)

        # Calculate the loss
        loss = loss_fn(X_fit, X)

        # Run backward pass
        loss.backward()

        # Optimize
        optimizer.step()
        optimizer.zero_grad()
        

    # Return the control points
    return P_tensor.detach()



def get_catmullrom_points(P_tensor, num_sample_pts):
    num_control_pts = P_tensor.shape[0]

    # Sample equally spaced points on the spline
    num_curves = num_control_pts - 1
    t = torch.linspace(0, num_control_pts - 1, num_sample_pts)

    # Add ghost points to make the spline pass through first and last point
    G0 = P_tensor[0] + (P_tensor[0] - P_tensor[1])
    G1 = P_tensor[-1] + (P_tensor[-1] - P_tensor[-2])
    P_extended = torch.vstack([G0, P_tensor, G1])

    # Get curve index for every t value and set the t value to between 0 and 1
    curve_indices = torch.clamp(t.floor().long(), 0, num_curves - 1)
    t = t - curve_indices
    # Get the bernstein coefficients
    t_val = torch.stack([
        0.5 * (-t + 2 * (t**2) - (t**3)),
        0.5 * (2 - 5 * (t**2) + 3 * (t**3)),
        0.5 * (t + 4 * (t**2) - 3 * (t**3)),
        0.5 * (-(t**2) + (t**3))
    ]).T

    # Get the control point coordinates
    p0 = P_extended[curve_indices]
    p1 = P_extended[curve_indices + 1]
    p2 = P_extended[curve_indices + 2]
    p3 = P_extended[curve_indices + 3]

    px = torch.stack([p0[:, 0], p1[:, 0], p2[:, 0], p3[:, 0]]).T
    py = torch.stack([p0[:, 1], p1[:, 1], p2[:, 1], p3[:, 1]]).T

    # Get the x and y coordinates of the sample points
    sample_x = torch.sum(t_val * px, axis = 1)
    sample_y = torch.sum(t_val * py, axis = 1)

    # Get the sample points
    X_fit = torch.stack([sample_x, sample_y]).T

    return X_fit


def curve_fit_loss(X_fit, X):
    with torch.no_grad():
        dists = torch.cdist(X, X_fit, p = 2)
        idx1 = torch.argmin(dists, dim = 1)
        dists[range(dists.size(0)), idx1] = float('inf')
        idx2 = torch.argmin(dists, dim = 1)
    

    # Get the altitude length
    # Get the two closest points from the sample points on spline to every point in the original
    X1 = X_fit[idx1]
    X2 = X_fit[idx2]

    # Get the altitude length
    h = get_altitude(X1, X2, X)
    
    loss = torch.mean(h * h)
    return loss


def get_altitude(X1, X2, X):
    a, b = X1[:, 0], X1[:, 1]
    c, d = X2[:, 0], X2[:, 1]
    e, f = X[:, 0], X[:, 1]

    # Get area of triangle formed by the three points
    A = torch.abs(a * d - b * c + c * f - d * e + b * e - a * f)
    # Get base width
    eps = 1e-10
    w = torch.sqrt((c - a) ** 2 + (d - b) ** 2 + eps)

    return A / w


def generate_airfoil_parameterization(airfoil_set: str, num_control_pts: int, num_sample_pts: int):
    # Load the airfoils
    data = np.load(f'generated_airfoils/{airfoil_set}/original_coordinates.npz')
    X_orig = data['X']

    total_airfoils = X_orig.shape[0]

    
    # Create dummy array to hold parametrized airfoils
    P_all = np.zeros((total_airfoils, num_control_pts * 2))

    # Create dummy array to hold parameterized airfoil L by D ratios
    L_by_D_all = np.zeros(total_airfoils)


    # For each airfoil fit a spline and store the parameterization in X_all
    for i in range(total_airfoils):
        X = X_orig[i, :].reshape(-1, 2)

        # Shift centroid to origin
        X_centroid = np.mean(X, axis = 0)
        X = X - X_centroid

        # Fit spline and get the control points
        P_tensor = fit_catmullrom(X.flatten(), num_control_pts)
        
        # Store the control points in the X_all array
        P_all[i, :] = P_tensor.numpy().flatten()

        # Compute the L by D ratio of the airfoil
        X_fit = get_catmullrom_points(P_tensor, num_sample_pts).numpy()
        L_by_D = compute_L_by_D(X_fit.flatten())
        L_by_D_all[i] = L_by_D

        if i % (total_airfoils // 20) == 0 or i == total_airfoils - 1:
                print_progress_bar(iteration = i, total_iterations = total_airfoils)
    

    # Save the airfoils and their L by D ratios to file
    save_filename = f'generated_airfoils/{airfoil_set}/original'
    np.savez(save_filename, P = P_all, L_by_D = L_by_D_all)