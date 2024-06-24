import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Parameters
train_batch_size = 1024
test_batch_size = 64
num_classes = 3
num_epochs = 200
input_channels = 3
input_height = 32
input_width = 32
hidden_size = 100

# Generate random training data
x_train = torch.randn(train_batch_size, input_channels,
                      input_height, input_width)
y_train = torch.randint(0, num_classes, (train_batch_size,))

# Generate random test data
x_test = torch.randn(test_batch_size, input_channels,
                     input_height, input_width)
y_test = torch.randint(0, num_classes, (test_batch_size,))

# Create DataLoaders
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False)

# Define the model with Batch Normalization


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.fc1 = nn.Linear(
            input_channels * input_height * input_width, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.bn1(x)
        x = x.view(-1, input_channels * input_height *
                   input_width)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, labels in train_dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Plot the loss
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print(
        f'Your model predicted {correct} out of the total {total} test samples correctly.')
