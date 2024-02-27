# Example of target with class indices
import torch.nn as nn
import torch
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
target = torch.randn(3, 5)
print(target[:,:2].softmax(dim=1))
print(target[:,2:].softmax(dim=1))
print(target.softmax(dim=1))
output = loss(input, target)
output.backward()