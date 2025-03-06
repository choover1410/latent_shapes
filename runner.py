import matplotlib.image
from fourier3 import main as run
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os.path
import os

while True:
    now = str(datetime.now().timestamp())
    now = now.replace('.', '')

    filled_matrix, integral_y = run()
    matplotlib.image.imsave(f'data/srm_{now}.png', filled_matrix, cmap='gray')

    df = pd.DataFrame(integral_y)
    df.to_csv(f'data/y_{now}.csv')