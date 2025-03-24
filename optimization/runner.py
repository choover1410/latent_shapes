import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from utils.sample_input import points as sample_points
from utils.parameterization import fit_catmullrom, get_catmullrom_points

# Import pre-trained network - Trained network must already exist!
from neural_network.trained_networks import burn_network


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize random input that is to be optimized
input_points = sample_points

# Normalize the shape
X_centroid = np.mean(input_points, axis = 0)
normalized_points = input_points - X_centroid

# Fit coordinates to Catmull-Rom spline and then torch it
X = fit_catmullrom(normalized_points.flatten(), num_control_pts = 12)
X = X.to(torch.float32).to(device)
X = X.reshape(-1, 2)

# We need to hold constant the first and last points, so make a new set
X_list = []
for i in range(X.shape[0]):
    X_list.append(X[i])
# Extract the portion of X to be optimized, leave the first and the last point
X_opt = torch.vstack(X_list[1:-1])
X_opt.requires_grad = True

# Combine the endpoints and optimizable middle points
X = torch.vstack([X_list[0], X_opt, X_list[-1]]).reshape(1, -1)

# TARGET
Y = torch.tensor([75.0]).reshape(1, -1).to(device)

# Initialize network architecture
burn_network = burn_network.to(device)

# Define the loss function
MSELoss_fn = nn.MSELoss()

## Define an optimizer
learning_rate = 0.0001
weight_decay = 0
optimizer = torch.optim.Adam([X_opt], lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

# Set the training properties
epochs = 500
print_cost_every = 1

# Set model to evaluation model to allow batch normalization to work with single input
burn_network.eval()


# Training Loop:
for epoch in range(1, epochs + 1):
    # Reconstruct input X after every gradient step
    X = torch.vstack([X_list[0], X_opt, X_list[-1]]).reshape(1, -1)

    # Run the forward pass and calculate the prediction
    with torch.set_grad_enabled(True):
        Y_pred = burn_network(X)

        # Compute the loss
        loss = MSELoss_fn(Y_pred, Y)

    # Run the backward pass and calculate the gradients
    loss.backward()

    # Take an update step and then zero out the gradients
    optimizer.step()
    optimizer.zero_grad()

    # Print training progress
    if epoch % print_cost_every == 0 or epoch == 1:
        J_train = loss.item()

        # Compute L by D predicted by Xfoil
        X_fit = get_catmullrom_points(X.detach().reshape(-1, 2), num_sample_pts = 201).detach().numpy()

        # Print the current performance
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch:{num_digits}}/{epochs}]. Train Cost: {J_train:11.6f}. Y: {Y.item():.2f}. Y_pred: {Y_pred.item():.2f}. X: {X_fit[:, 0].tolist()}.')

    
    scheduler.step(J_train)

# Get coordinates on the boundary
X_fit = get_catmullrom_points(X.detach().reshape(-1, 2), num_sample_pts = 201)

# Plot the final result
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.axis('equal')
plt.savefig('Final_airfoil.png', dpi = 600)