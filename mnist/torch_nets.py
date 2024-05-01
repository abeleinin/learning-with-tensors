import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Net1(nn.Module):
  def __init__(self):
    super(Net1, self).__init__()
    self.layer = nn.Linear(28*28, 10)
  
  def forward(self, x):
    x = F.relu(self.layer(x))
    return x

class Net2(nn.Module):
  def __init__(self):
    super(Net2, self).__init__()
    self.layer1 = nn.Linear(28*28, 512)
    self.layer2 = nn.Linear(512, 10)
  
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return x

class Net3(nn.Module):
  def __init__(self):
    super(Net3, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    self.maxpool1 = nn.MaxPool2d(3)
    self.conv2 = nn.Conv2d(32, 64, 2)
    self.maxpool2 = nn.MaxPool2d(2)
    self.dropout = nn.Dropout(0.25)
    self.fc1 = nn.Linear(4*4*64, 4*4*64)
    self.fc2 = nn.Linear(4*4*64, 10)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = self.maxpool(out)
    out = F.relu(self.conv2(out))
    out = out.reshape(out.size(0), -1)
    out = self.dropout(out)
    out = self.fc1(out)
    out = self.dropout(out)
    out = self.fc2(out)
    return x

model = Net3()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 32

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

epochs = 20
for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  for i, (images, labels) in enumerate(train_loader, 0):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i % 100 == 99:
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/100:.4f}')
      running_loss = 0.0

# Save model
torch.save(model.state_dict(), 'weights/net3.pth')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

