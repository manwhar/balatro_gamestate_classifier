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
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(5, 6)
        self.layer2 = nn.Linear(6, 8)
        self.layer3 = nn.Linear(8, 10)
        self.layer4 = nn.Linear(10, 15)

    def forward(self, input):
        partial = self.relu(self.layer1(input))
        partial = self.relu(self.layer2(partial))
        partial = self.relu(self.layer3(partial))
        partial = self.relu(self.layer4(partial))
        model_output = self.sigmoid(self.layer3(partial))
        return model_output


LEARNING_RATE = 100
EPOCHS = 10000

model = myNN()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


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
