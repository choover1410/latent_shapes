import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from optimization.utils.sample_input2 import x_points as sample_points
from optimization.utils.parameterization import fit_catmullrom, get_catmullrom_points
from optimization.checkpoints.net_load import burn_network


y_points = [0.41467085, 0.41483114, 0.41838077, 0.420449, 0.42491956, 0.43108353,
    0.43998984, 0.44097028, 0.45290862, 0.46464842, 0.47677692, 0.50906301,
    0.52226878, 0.53532645, 0.54856748, 0.56186702, 0.57496355, 0.58844849,
    0.60203725, 0.61568138, 0.62954352, 0.64341272, 0.65738954, 0.67145767,
    0.68550832, 0.69967037, 0.71384532, 0.72803857, 0.74227122, 0.75653205,
    0.77079719, 0.78508297, 0.79939678, 0.8136982, 0.82804055, 0.84237627,
    0.85670621, 0.87108935, 0.88543467, 0.89980128, 0.91419061, 0.9285569,
    0.94294632, 0.95733567, 0.97172386, 0.9945424, 0.16053395, 0.06926921,
    0.02503454, 0.0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Initialize random input that is to be optimized
input_points = sample_points

# Fit coordinates to Catmull-Rom spline and then torch it
X = fit_catmullrom(input_points.flatten(), num_control_pts = 150)
X_fit = get_catmullrom_points(X.detach().reshape(-1, 2), num_sample_pts = 201).detach().numpy()
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.axis("equal")
plt.show()

X = X.to(torch.float32).to(device)
X = X.reshape(1, -1)
X.requires_grad = True

# TARGET
y_tens = torch.full((1, 50), 0.85)
Y = torch.tensor(y_tens).reshape(1, -1).to(device)

# Initialize network architecture
burn_network = burn_network.to(device)

# Define the loss function
MSELoss_fn = nn.MSELoss()

## Define an optimizer
learning_rate = 0.0001
learning_rate = 0.01

weight_decay = 0
optimizer = torch.optim.Adam([X], lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

# Set the training properties
epochs = 500
print_cost_every = 2

# Set model to evaluation model to allow batch normalization to work with single input
burn_network.eval()


fig1 = plt.figure()
fig2 = plt.figure()
# Training Loop:
for epoch in range(1, epochs + 1):
    # Reconstruct input X after every gradient step

    # Run the forward pass and calculate the prediction
    with torch.set_grad_enabled(True):
        Y_pred = burn_network(X)

        # Compute the loss
        loss = MSELoss_fn(Y_pred, Y)
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

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
        #plt.clf()
        #plt.plot(X_fit[:, 0], X_fit[:, 1])
        #plt.axis("equal")
        #plt.draw()
        #plt.pause(0.01)  # Pause to update the plot

        plt.cla()
        plt.plot(y_points)
        plt.plot(Y_pred.detach().numpy().flatten())
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

    scheduler.step(J_train)

# Get coordinates on the boundary
X_fit = get_catmullrom_points(X.detach().reshape(-1, 2), num_sample_pts = 201)

# Plot the final result
plt.figure()
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.show()
plt.axis('equal')
plt.savefig('Final_airfoil.png', dpi = 600)

Y_final = burn_network(X)
plt.figure()
plt.plot(y_points)
plt.plot(Y_final.detach().numpy().flatten())
plt.title("Burnback")
plt.show()
