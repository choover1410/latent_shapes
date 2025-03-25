import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from optimization.utils.sample_input import points as sample_points
from optimization.utils.parameterization import fit_catmullrom, get_catmullrom_points
from optimization.checkpoints.net_load import burn_network


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Initialize random input that is to be optimized
input_points = sample_points

# Fit coordinates to Catmull-Rom spline and then torch it
X = fit_catmullrom(input_points.flatten(), num_control_pts = 150)
X = X.to(torch.float32).to(device)
X = X.reshape(1, -1)
X.requires_grad = True

# TARGET
y_tens = torch.full((1, 50), 0.77)
Y = torch.tensor(y_tens).reshape(1, -1).to(device)

# Initialize network architecture
burn_network = burn_network.to(device)

# Define the loss function
MSELoss_fn = nn.MSELoss()

## Define an optimizer
learning_rate = 0.0001
learning_rate = 0.0101

weight_decay = 0
optimizer = torch.optim.Adam([X], lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

# Set the training properties
epochs = 10000
print_cost_every = 1

# Set model to evaluation model to allow batch normalization to work with single input
burn_network.eval()


# Training Loop:
for epoch in range(1, epochs + 1):
    # Reconstruct input X after every gradient step

    # Run the forward pass and calculate the prediction
    with torch.set_grad_enabled(True):
        Y_pred = burn_network(X)

        # Compute the loss
        loss = MSELoss_fn(Y_pred, Y)
        print(f"Loss: {loss.item()}")

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
        #print(f'Epoch: [{epoch:{num_digits}}/{epochs}]. Train Cost: {J_train:11.6f}. Y: {Y.item():.2f}. Y_pred: {Y_pred.item():.2f}. X: {X_fit[:, 0].tolist()}.')

    
    scheduler.step(J_train)

# Get coordinates on the boundary
X_fit = get_catmullrom_points(X.detach().reshape(-1, 2), num_sample_pts = 201)

# Plot the final result
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.show()
plt.axis('equal')
plt.savefig('Final_airfoil.png', dpi = 600)