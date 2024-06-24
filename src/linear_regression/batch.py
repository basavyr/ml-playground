import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

import torch.nn.functional as F

# Example data (ensure x_data and y_data have enough samples, e.g., 760 samples each)
x_data = np.random.rand(760, 1)
y_data = 3 * x_data + np.random.rand(760, 1)

# Split the data into training and test sets
x_train = x_data[:640]
y_train = y_data[:640]
x_test = x_data[640:]
y_test = y_data[640:]

# Hyperparameters
batch_size = 64
num_epochs = 100
learning_rate = 0.01

# Convert numpy arrays to torch tensors
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Create a dataset and data loader for training data
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define a simple linear regression model


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(1, 10)
        self.fc3 = nn.Linear(10, 850)
        self.bn3 = nn.BatchNorm1d(850)
        self.fc4 = nn.Linear(850, 50)
        self.fc5 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.tanh(x)
        # x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn3(self.fc3(x))
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        return x


model = LinearRegressionModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
train_losses = []
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
              1, num_epochs, loss.item()))


# Plot the loss function
plt.plot(train_losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.eval()  # Set the model to evaluation mode
eval_losses = []
with torch.no_grad():  # Disable gradient computation
    test_outputs = model(x_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    eval_losses.append(test_loss.item())
    print('Test Loss: {:.4f}'.format(test_loss.item()))


# Plot the graph
predicted = model(x_train_tensor).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Plot the test data
predicted_test = model(x_test_tensor).detach().numpy()
plt.plot(x_test, y_test, 'bo', label='Test data')
plt.plot(x_test, predicted_test, 'go', label='Test prediction')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
