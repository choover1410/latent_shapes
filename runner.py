import matplotlib.image
import matplotlib.pyplot as plt
from fourier3 import main as fourier3
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os.path
import os
import torch


from PIL import Image

def scale_down_matrix(image, output_size):
    # Ensure output size is a tuple of (width, height)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Use interpolation to resize
    scaled_image = np.array(Image.fromarray(image).resize(output_size))
    return scaled_image


for i in range(10000):
    print(f"Burning object {i} of {10000}")
    now = str(datetime.now().timestamp())
    now = now.replace('.', '')

    filled_matrix, integral_y = fourier3()

    # Handle X
    int_filled_matrix = filled_matrix.astype(int)
    scaled_image = scale_down_matrix(int_filled_matrix, 1024)
    plt.imsave(f'data/x_{now}.png', scaled_image, cmap='gray')

    # Handle Y
    plt.imsave(f'data/y_{now}.png', integral_y, cmap='gray')
