import matplotlib.image
from fourier3 import main as fourier3
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os.path
import os
import torch

for i in range(10000):
    print(f"Burning object {i} of {10000}")
    now = str(datetime.now().timestamp())
    now = now.replace('.', '')

    filled_matrix, integral_y = fourier3()
    print(integral_y)

    int_filled_matrix = filled_matrix.astype(int)

    matplotlib.image.imsave(f'data/srm_{now}.png', filled_matrix, cmap='gray')


    with open(f'data/y_{now}.csv', 'w') as file:
        for i, num in enumerate(integral_y):
            if i < len(integral_y) - 1:
                file.write(str(num) + '\n')
            else:
                file.write(str(num))
