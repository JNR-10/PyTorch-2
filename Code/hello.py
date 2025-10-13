# Basic PyTorch example
import torch

# Create two tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Add the tensors
z = x + y

print('x:', x)
print('y:', y)
print('z = x + y:', z)
