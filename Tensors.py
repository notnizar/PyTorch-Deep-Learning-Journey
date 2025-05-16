import torch

# PyTorch Tensors
# Tensors repressent item in numerical data that can be processed by a computer
# Tensors are a generalization of matrices to higher dimensions

# Scalar is just a single number (0D tensor, "How much ?")
Scalar = torch.tensor(data=5)
print (Scalar)

# Vector can have multiple numbers (1D tensor, "How much ? and in which direction ?")
Vector = torch.tensor([1, 2, 3])
print (Vector)

# Matrix is a grid of numbers (2D tensor, its like a collision of vectors,)
MATRIX = torch.tensor([[1, 2, 3], [4, 5, 6]]) #Ressarch why MATRIX uppercase
print (MATRIX)

# Tensor is a flexible container that can hold data in n-dimensions (nD tensor)
TENSORS = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print (TENSORS)


# Some of Tensor Attributes
# .item() is a method of the tensor that returns the value of the tensor as a standard Python number

# .shape is a property of the tensor that returns the size of each dimension
# .ndim() is a method of the tensor that returns the number of dimensions

# .dtype is a property of the tensor that returns the data type of the tensor
# .device is a property of the tensor that returns the device of the tensor (CPU or GPU)



# Random tensors
RandomTensor = torch.rand(2, 3) # We use random tensors to initialize the weights of a neural network then adjast them during training
print (RandomTensor)

# Imagine we want to present a wether forecast for 24 days, 10 diffrent location, with 4 features (temperature, humidity, wind speed, and precipitation) for each day
RandomTensor = torch.rand(24, 10, 4)
print (RandomTensor)


# Zeros and Ones tensors have many uses in machine learning, such a creating masks like we want to ignore some data
# Create a tensor of all zeros
ZerosTensor = torch.zeros(2, 3)
print (ZerosTensor)

# Create a tensor of all ones
OnesTensor = torch.ones(2, 3)
print (OnesTensor)



# .arange() creates a 1D tensor with a range of values there is 2 ways to use it
# 1. By passing the start and end values and the step size
RangeTensor_1 = torch.arange(start=0, end=10, step=2) #start is optional and default is 0, end is required and step is optional and default is 1
# Stars is optional and default is 0,
# end is required and step is optional and default is 1
# Step is optional and default is 1, step is the difference between each value in the range like 0, 2, 4, 6, 8 every time we add 2


# 2. Dirctly passing the end value
RangeTensor_2 = torch.arange(10)
print (RangeTensor_1)


# data types
# PyTorch supports many data types, including:

# 1. torch.float32 (32-bit floating point)
float_32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print (float_32)
# 2. torch.float64 (64-bit floating point)
float_64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
print (float_64)
# there are many other data types like int, bool, etc see the documentation for more https://docs.pytorch.org/docs/stable/tensors.html

tensor_1 = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor_2 = torch.tensor([4, 5, 6], dtype=torch.float32)

# Tensors operations
# 1. Addition
torch.add(2, 3) # 2 + 3 = 5
# 2. Subtraction
torch.sub(2, 3) # 2 - 3 = -1
# 3. Multiplication
torch.mul(2, 3) # 2 * 3 = 6
# 4. Division
torch.div(2, 3) # 2 / 3 = 0.6666666666666666
# 5. Exponentiation
torch.pow(2, 3) # 2 ** 3 = 8
# 6. Square root
torch.sqrt(4) # sqrt(4) = 2.0
# 7. Absolute value
torch.abs(-2) # abs(-2) = 2
# and many more


# there is 2 main way in multiplying tensors
# 1. Element wise multiplication
torch.mul(tensor_1, tensor_2) # [1*4, 2*5, 3*6] = [4, 10, 18]
# 2. Matrix multiplication
torch.matmul(tensor_1, tensor_2) # [1*4 + 2*5 + 3*6] = [32]

# important note: inner dimension of the first tensor must be equal to the outer dimension of the second tensor or it will not work
# example:
tensor_1 = torch.tensor([[1, 2]])
tensor_2 = torch.tensor([[1, 2]])
torch.matmul(tensor_1, tensor_2) # error

# So if you want to solve this problem you can use the transpose method (. T) to transpose any of the tensors
tensor_1 = torch.tensor([[1, 2]])
tensor_2 = torch.tensor([[1, 2]])
torch.matmul(tensor_1, tensor_2.T) # [1*1 + 2*2] = [5]

# finding the min, max, mean, and sum of a tensor
tensor = torch.tensor([[1, 2], [3, 4]])
print (tensor.min()) # 1
print (tensor.max()) # 4
torch.mean(tensor.type(torch.float32)) # 2.5 (mean should be float "torch.float32")
print (tensor.sum()) # 10

# now there is the .argmax() and .argmin() methods that return the index of the min and max values
tensor_arg = torch.tensor([[1, 2, 3, 4]])
tensor.argmax() # 3
tensor.argmin() # 0



# how to change the stracture of a tensor (reshape, view, flatten, squeeze, unsqueeze):

# 1. Reshape
# Reshape the tensors, reshaping is the process of changing the shape of a tensor without changing its data
# ex: Reshape the tensor to (2, 4)
reshaped_tensor_1 = tensor.reshape(2, 4)
# the total number of elements in the tensor must be the same before and after reshaping
print (reshaped_tensor_1) # [[1, 2, 3, 4], [0, 0, 0, 0]]

# 2. View
# View is similar to reshape, but it returns a view of the original tensor, not a copy
# ex: View the tensor as a 1D tensor
view_tensor = tensor.view(-1)
# -1 means that the size of this dimension is inferred from the other dimensions
print (view_tensor) # [1, 2, 3, 4]

# note: view and reshape are similar, but view is faster because it does not create a copy of the tensor Search for the difference between view and reshape in pytorch

# 3. Stack
# Stack is used to stack multiple tensors along a new dimension
# ex: Stack the tensors along a new dimension
tensor_1 = torch.tensor([[1, 2], [3, 4]])
tensor_2 = torch.tensor([[5, 6], [7, 8]])
stacked_tensor = torch.stack((tensor_1, tensor_2), dim=0)
# the dim parameter specifies the dimension along which to stack the tensors
print (stacked_tensor) # [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

# 4. Squeeze
# Squeeze is used to remove dimensions of size 1 from the tensor
# ex: Squeeze the tensor
squeezed_tensor = torch.tensor([[1], [2], [3]])
squeezed_tensor = squeezed_tensor.squeeze()
print (squeezed_tensor) # [1, 2, 3]

# 5. Unsqueeze
# Unsqueeze is used to add a dimension of size 1 to the tensor
# ex: Unsqueeze the tensor
unsqueezed_tensor = torch.tensor([1, 2, 3])
unsqueezed_tensor = unsqueezed_tensor.unsqueeze(0)
print (unsqueezed_tensor) # [[1, 2, 3]]

# 6. Flatten
# Flatten is used to flatten the tensor to a 1D tensor
# ex: Flatten the tensor
flattened_tensor = torch.tensor([[1, 2], [3, 4]])
flattened_tensor = flattened_tensor.flatten()
print (flattened_tensor) # [1, 2, 3, 4]
