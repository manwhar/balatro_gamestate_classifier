import torch
import torch.nn as nn

predicted1 = torch.tensor([340.0])
predicted2 = torch.tensor([-1.2])
target = torch.tensor([0.0])

loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(predicted1, target)

print(loss)
