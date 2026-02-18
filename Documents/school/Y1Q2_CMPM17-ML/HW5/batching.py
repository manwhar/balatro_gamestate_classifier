import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 15)
        self.layer2 = nn.Linear(15, 5)
        self.layer3 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.layer1(input))
        x = self.relu(self.layer2(x))
        output = self.layer3(x)
        return output


#### DATASET CODE ####
df = pd.read_csv("color_dataset.csv")
data = torch.tensor(df.values, dtype=torch.float)

# Change to have the first 800 data points as train, the next 300 as validation, and the rest as test
train_inputs = data[:800, 0:3]
val_inputs = data[800:1100, 0:3]
test_inputs = data[1100:, 0:3]

train_outputs = data[:800, 3]
val_outputs = data[800:1100, 3]
test_outputs = data[1100:, 3]
#### END DATASET CODE ####

scaler = StandardScaler()
scaler.fit(train_inputs)
scaled_train_inputs = torch.tensor(scaler.transform(train_inputs), dtype=torch.float)
scaled_val_inputs = torch.tensor(scaler.transform(val_inputs), dtype=torch.float)
scaled_test_inputs = torch.tensor(scaler.transform(test_inputs), dtype=torch.float)


class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        ### Get the input and output at index idx
        return self.inputs[idx], self.outputs[idx]


train_set = MyDataset(train_inputs, train_outputs)
val_set = MyDataset(val_inputs, val_outputs)
test_set = MyDataset(test_inputs, test_outputs)

train_loader = DataLoader(train_set, batch_size=32)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

model = MyModel()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
NUM_EPOCHS = 20
### Create Dataset objects for train, val, and test, and create DataLoaders with these objects

# training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for x_batch, y_batch in train_loader:
        ### Get inputs and outputs in batches using the training DataLoader
        train_preds = model(x_batch)
        loss = criterion(train_preds, y_batch.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Calculate the batch accuracy for training (see Week 5 Day 2 slides for reminder!)
        train_accuracy = 0
        print(f"Epoch {epoch} | Loss: {loss.item()} | Accuracy: {train_accuracy}")

    ### Include loop for validation dataset here.
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            val_preds = model(x_batch)
            loss = criterion(val_preds, y_batch.unsqueeze(1))

            val_accuracy = 0
            print(f"Val loss: {loss.item()} | Accuracy: {val_accuracy}")


print("\n------------------------Testing Phase-----------------------------\n")
model.eval()
with torch.no_grad():
    ### Get inputs and outputs in batches using the testing DataLoader
    for x_batch, y_batch in test_loader:
        test_preds = model(x_batch)
        loss = criterion(test_preds, y_batch.unsqueeze(1))

        ### Calculate the batch accuracy for testing (see Week 5 Day 2 slides for reminder!)
        class_preds = test_preds > 0
        num_correct = (class_preds.squeeze() == y_batch).sum()
        test_accuracy = num_correct / len(y_batch)

        print(f"Loss: {loss.item()} | Accuracy: {test_accuracy}")


### Calculate precision and recall for testing set and display results with a confusion matrix

# Try out your model on diffeent RGB values! Plug them in (with a decimal point) and see if they make sense.
print(model(torch.tensor([50, 168, 82.0])))
