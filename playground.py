import torch


tensor_1 = torch.tensor([[1, 2], [3, 4]])
tensor_2 = torch.tensor([[5, 6], [7, 8]])
stacked_tensor = torch.stack((tensor_1, tensor_2), dim=1)
# the dim parameter specifies the dimension along which to stack the tensors
print (stacked_tensor)