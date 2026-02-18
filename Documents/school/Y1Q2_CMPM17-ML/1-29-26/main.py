import torch.nn as nn

import pandas as pd
import torch

df = pd.read_csv(
    r"C:\Users\thatp\Documents\school\Y1Q2_CMPM17-ML\1-29-26\fake_data.csv"
)
print(df.info())
print(df.head())

data = torch.tensor(df.values, dtype=torch.float)


inputs = data[:, 0:4]
outputs = data[:, 4:8]

train_in = inputs[: len(inputs // 1.5), :]
train_out = outputs[: len(outputs / 1.5), :]

test_in = inputs[len(inputs // 1.5) : len(inputs), :]
test_out = outputs[len(inputs // 1.5) : len(inputs), :]


class myNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(4, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 4)

    def forward(self, input):
        partial = self.relu(self.layer1(input))
        partial = self.relu(self.layer2(partial))
        model_output = self.relu(self.layer3(partial))
        return model_output


model = myNN()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.15)

EPOCHS = 2000

# training loop
for i in range(EPOCHS):
    pred = model(inputs)
    loss = loss_function(pred, outputs)

    loss.backward()  # calculate slopes to guide optimizer
    optimizer.step()
    optimizer.zero_grad()  # reset optimizer slopes for next step

# testing "loop"
with torch.no_grad():
    print("TESTING:")
    pred = model(inputs)
    loss = loss_function(pred, outputs)
    print(loss.item())

torch.save(model.state_dict(), "model.pt")

# model.load_state_dict(torch.load("model.py", weights_only=True))
