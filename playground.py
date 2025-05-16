import torch
import numpy as np

numpy_array = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(numpy_array).to('cuda')
print (tensor) # [[1, 2], [3, 4]]