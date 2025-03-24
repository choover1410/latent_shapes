import os.path
import time
import numpy as np
import torch
from torch import nn
from neural_network.network_definition import NeuralNetwork
from utils import train_loop, dev_loop

"""
Training Set Coordinates -> Training Set Catmull-Rom Spline Params
Training Set Catmull-Rom Spline Params -> Noised Training Set Catmull-Rom Spline Params
Combine all sets
Filter any garbage
Permute all sets
...

Must be numpy arrays with arrays:
    - "P" for the input
    - "L_by_D" for the output

[
    "P": [
        [X_data_1],
        [X_data_2],
        ...
    ],
    "L_by_D": [
        [Y_data_1],
        [Y_data_2],
        ...
    ]
]

"""

# Get the training data
train_filepath = 'data/train_data.npz'
data_train = np.load(train_filepath)

# Get the validation data
dev_filepath = 'data/validate_data.npz'
data_dev = np.load(dev_filepath)

# cuda cuda cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Arbitrary count to max out the training data
count = 1348100 * 100

# Massage training data X
X_train = torch.from_numpy(data_train['X']).to(torch.float32)[:12].to(device)
print(f"Shape of X_train: {X_train.shape}")

# Massage training data Y
Y_train = torch.from_numpy(data_train['Y']).to(torch.float32).to(device)
print(f"Shape of X_train: {Y_train.shape}")

# Massage validation data X
X_val = torch.from_numpy(data_dev['X']).to(torch.float32).to(device)
print(f"Shape of X_val: {X_val.shape}")

# Massage validation data Y
Y_val = torch.from_numpy(data_dev['Y']).to(torch.float32).to(device)
print(f"Shape of Y_Val: {Y_val.shape}")


# Initialize the burn network (to be trained)
torch.manual_seed(0)
input_dim, hidden_dim, layer_count = 300, 300, 10
burn_network = NeuralNetwork(input_dim, hidden_dim, layer_count).to(device)


# Make changes for running the computation faster
compute_optimizations = False
if compute_optimizations == True:
    try:
        burn_network = torch.compile(burn_network)
    except:
        print('Could not compile the network.')
    torch.set_float32_matmul_precision('high')


# Define the loss function
MSELoss_fn = nn.MSELoss()


# Define an optimizer
learning_rate = 0.01
weight_decay = 1e-4
optimizer = torch.optim.Adam(burn_network.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)


# Set the training properties
epochs = 5000
print_cost_every = 1
B_train = X_train.shape[0] // 10 # batch size for training data
B_train = 2
B_val = X_val.shape[0]  # batch size for validation data (must be 1 for validation data)


# Load saved model if available
if os.path.exists('checkpoints/latest.pth'):
    checkpoint = torch.load('checkpoints/latest.pth')
    burn_network.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    total_epochs = checkpoint['total_epochs']
else:
    total_epochs = 0


# Train the burn network
burn_network.train()
for epoch in range(total_epochs + 1, total_epochs + epochs + 1):

    verbose = True if epoch % print_cost_every == 0 or epoch == total_epochs + 1 else False
    save = verbose

    # logging shenanigans
    if verbose:
        print(f"Epoch {epoch}\n" + 40 * '-')
        print(scheduler.get_last_lr())
        t0 = time.perf_counter()

    # Run the training loop
    J_train = train_loop(X_train, Y_train, B_train, burn_network, MSELoss_fn, optimizer, verbose = verbose, compute_optimizations=compute_optimizations)

    # Run the validation loop
    J_val = dev_loop(X_val, Y_val, B_val, burn_network, MSELoss_fn, verbose, compute_optimizations=compute_optimizations)
    scheduler.step(J_val)

    # More logging shenanigans
    if verbose:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000 # Time difference in milliseconds
        print(f'Time taken: {dt}')

    if save:
        # Create checkpoint and save the model
        checkpoint = {
            'total_epochs': epoch,
            'model': burn_network.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # Save the model twice: once on its own and once in the latest model file
        torch.save(checkpoint, f'checkpoints/burn_network_Epoch_{epoch}_Jtrain{J_train:.3e}_Jval_{J_val:.3e}.pth')
        torch.save(checkpoint, f'checkpoints/latest.pth')

print('Finished Training!')