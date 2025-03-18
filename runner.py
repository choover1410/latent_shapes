import matplotlib.image
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

    int_filled_matrix = filled_matrix.astype(int)
    
    # Example usage
    scaled_image = scale_down_matrix(int_filled_matrix, 512)

    """
    matplotlib.image.imsave(f'data/srm_{now}.png', scaled_image, cmap='gray')

    with open(f'data/y_{now}.csv', 'w') as file:
        for i, num in enumerate(integral_y):
            if i < len(integral_y) - 1:
                file.write(str(num) + '\n')
            else:
                file.write(str(num))"
    """
