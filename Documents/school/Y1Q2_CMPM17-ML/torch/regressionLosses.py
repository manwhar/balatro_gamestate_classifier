import torch
import torch.nn as nn

loss_fn = nn.MSELoss()

predicted_values = torch.tensor([1.0, -2.0, 3.0])
actual_values = torch.tensor([2.0, 3.0, 1.0])
loss = loss_fn(predicted_values, actual_values)
print(loss)
