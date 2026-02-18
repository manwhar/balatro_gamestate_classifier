import torch
import torch.nn as nn

data_lst = 

data = torch.Tensor([[1, -2], [0, 0], [1, 3], [-9, -9]])

print(data)

relu = nn.ReLU()
layer1 = nn.Linear(2, 3)  # 2 inputs; 3 outputs
layer2 = nn.Linear(3, 4)
layer3 = nn.Linear(4, 2)

output1: torch.Tensor = relu(layer1(data))
output2: torch.Tensor = relu(layer2(output1))
output3: torch.Tensor = layer3(output2)

loss = ...
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(output3.tolist())
print(layer3.weight)
