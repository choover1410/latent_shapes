import matplotlib.pyplot as plt
import numpy as np
import torch
from parameterization import fit_catmullrom, get_catmullrom_points
from sample_input import points


P_tensor = fit_catmullrom(points, 200)
print(P_tensor)

X_fit = get_catmullrom_points(P_tensor.detach().reshape(-1, 2), num_sample_pts = 201).detach().numpy()
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.axis("equal")
plt.show()