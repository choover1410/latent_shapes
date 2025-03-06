from burn import burn_grain
import random
import numpy as np
import pylab as plt
import skfmm

N=10
W = 1001
X, Y = np.meshgrid(np.linspace(-1.1,1.1,W), np.linspace(-1.1,1.1,W))

# Create initial geometry
phi = 0 * np.ones_like(X)
phi[(X**2 + Y**2) > 0.25] = 1
phi[np.logical_and(np.abs(X) < 0.2, Y> -0.65)] = 0
#phi[np.logical_and(np.abs(X) < 0.75, np.abs(Y) < 0.05)] = 0

integral_y, d = burn_grain(X,Y,phi,N,1e-2)



# Create initial geometry
phi = 0 * np.ones_like(X)
phi[(X**2 + Y**2) > 0.25] = 1
#phi[np.logical_and(np.abs(X) < 0.2, Y> -0.65)] = 0
phi[np.logical_and(np.abs(X) < 0.75, np.abs(Y) < 0.05)] = 0

integral_y_2, d = burn_grain(X,Y,phi,N,1e-2)


plt.figure()
plt.plot([i/len(integral_y) for i,_ in enumerate(integral_y)], integral_y)
plt.plot([i/len(integral_y_2) for i,_ in enumerate(integral_y_2)], integral_y_2)
plt.legend(["Nominal","Actual", "Actual2"])
plt.show()
