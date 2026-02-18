import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import torch

# pd.set_option("display.max_rows", None)

df = pd.read_csv(
    r"C:\Users\thatp\Documents\school\Y1Q2_CMPM17-ML\midterm\housing_dataset.csv"
)

df = df.dropna()
df = df.drop_duplicates()

df["ocean_proximity"] = df["ocean_proximity"].str.lower()


df = df[df["total_rooms"] < 75000]  # drop total room outliers
df = pd.get_dummies(
    df, dtype=float, columns=["ocean_proximity"]
)  # turn ocean proximity into onehots


# split data into test/train/val

# length = 20432


class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, _id):
        return self.inputs[_id], self.outputs[_id]

    def __len__(self):
        return len(self.inputs)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(13, 30)
        self.layer2 = nn.Linear(30, 50)
        self.layer3 = nn.Linear(50, 30)
        self.layer4 = nn.Linear(30, 10)
        self.layer5 = nn.Linear(10, 1)

    def forward(self, input):
        partial = self.relu(self.layer1(input))
        partial = self.relu(self.layer2(partial))
        partial = self.relu(self.layer3(partial))
        partial = self.relu(self.layer4(partial))
        output = self.layer5(partial)
        return output


inputs = df[
    [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity_<1h ocean",
        "ocean_proximity_inland",
        "ocean_proximity_island",
        "ocean_proximity_near bay",
        "ocean_proximity_near ocean",
    ]
]
outputs = df["median_house_value"]

print(inputs.head())
print(outputs.head())

train_inputs = torch.tensor(inputs[:15500].values, dtype=torch.float)
# val_inputs = torch.tensor(inputs[13200:15500].values, dtype=torch.float)
test_inputs = torch.tensor(inputs[15500:].values, dtype=torch.float)

train_outputs = torch.tensor(outputs[:15500].values, dtype=torch.float).unsqueeze(1)
# val_outputs = torch.tensor(outputs[13200:15500].values, dtype=torch.float)
test_outputs = torch.tensor(outputs[15500:].values, dtype=torch.float).unsqueeze(1)

train_data = MyDataset(train_inputs, train_outputs)
test_data = MyDataset(test_inputs, test_outputs)

BATCH_SIZE = 200
# batch size 200, shuffle=true
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# loop through init/train/test ITERS times to find the best seed for optimizing loss
# (model never got below 51k and i got tired so. never finished this i guess)
ITERS = 1
best_loss = float("inf")
for j in range(ITERS):
    torch.manual_seed(j)
    model = Net()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 150
    # training loop
    for i in range(EPOCHS):
        rolling_loss = 0.0
        for inputs, outputs in train_dataloader:
            pred = model(inputs)
            loss = loss_function(pred, outputs)
            rmse_loss = torch.sqrt(loss)

            loss.backward()  # calculate slopes to guide optimizer
            optimizer.step()
            optimizer.zero_grad()  # reset optimizer slopes for next step
            rolling_loss += rmse_loss.item() * BATCH_SIZE  # type:ignore
        rolling_loss /= len(train_dataloader.dataset)  # type:ignore
        print(f"EPOCH {i}: l={rolling_loss:.4f}")

    # test
    with torch.no_grad():
        rolling_loss = 0.0
        for inputs, outputs in test_dataloader:
            pred = model(inputs)
            loss = loss_function(pred, outputs)
            rmse_loss = torch.sqrt(loss)

            rolling_loss += rmse_loss.item() * BATCH_SIZE
        rolling_loss /= len(test_dataloader.dataset)  # type:ignore
        print(f"TEST LOSS: {rolling_loss}")
    if rolling_loss < best_loss:
        best_loss = rolling_loss
        best_seed = j

print(
    f"Finished {ITERS} trials with best loss of {best_loss} found using seed {best_seed}."
)
