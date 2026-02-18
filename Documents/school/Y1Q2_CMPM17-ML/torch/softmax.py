import torch
import torch.nn as nn

logits = torch.tensor([1, 0.5, -0.1])
# logits don't yet add up to one

softmax = nn.Softmax(dim=0)
probabilities = softmax(logits)

print(probabilities)
