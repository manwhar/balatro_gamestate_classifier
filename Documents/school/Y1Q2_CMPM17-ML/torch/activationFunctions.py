import torch
import torch.nn as nn

const_tensor = torch.tensor(7, dtype=torch.float)
line_tensor = torch.tensor([1, -2, 3], dtype=torch.float)
flat_tensor = torch.tensor([[4, 5, 6], [-7, 8, -9], [0, 0, 1]], dtype=torch.float)
relu = nn.ReLU()

print(f"ReLU const: {relu(const_tensor)}")
print(f"ReLU line: {relu(line_tensor)}")
print(f"ReLU flat: {relu(flat_tensor)}")
