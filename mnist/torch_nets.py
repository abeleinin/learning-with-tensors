import argparse
import time

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
    out = x.view(-1, 28*28)
    out = F.relu(self.layer(out))
    return out

class Net2(nn.Module):
  def __init__(self):
    super(Net2, self).__init__()
    self.layer1 = nn.Linear(28*28, 512)
    self.layer2 = nn.Linear(512, 10)
  
  def forward(self, x):
    out = x.view(-1, 28*28)
    out = F.relu(self.layer1(out))
    out = F.relu(self.layer2(out))
    return out

class Net3(nn.Module):
  def __init__(self):
    super(Net3, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    self.maxpool1 = nn.MaxPool2d(3) # 8 x 8 x 32
    self.conv2 = nn.Conv2d(32, 64, 1)
    self.maxpool2 = nn.MaxPool2d(2) # 4 x 4 x 64
    self.dropout = nn.Dropout(0.25)
    self.fc1 = nn.Linear(4*4*64, 4*4*64)
    self.fc2 = nn.Linear(4*4*64, 10)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = self.maxpool1(out)
    out = F.relu(self.conv2(out))
    out = self.maxpool2(out)
    out = out.reshape(out.size(0), -1)
    out = self.dropout(out)
    out = self.fc1(out)
    out = self.dropout(out)
    out = self.fc2(out)
    return out

def main(args):
  batch_size = args.batch
  num_epochs = 20
  learning_rate = 1e-3

  if args.net == 1:
    model = Net1()
  elif args.net == 2:
    model = Net2()
  elif args.net == 3:
    model = Net3()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

  transform = transforms.Compose([
      transforms.ToTensor(), 
      transforms.Normalize((0.5,), (0.5,))
  ])

  train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

  train_size = int(0.7 * len(train_dataset))
  val_size = len(train_dataset) - train_size
  train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
  test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  for epoch in range(num_epochs):
    tic = time.perf_counter()
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
        toc = time.perf_counter()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/100:.4f}, Time {toc - tic:.3f} (s)')
        tic = time.perf_counter()
        running_loss = 0.0
    scheduler.step()
    model.eval()
    with torch.no_grad():
      validation_loss = sum(criterion(model(data.to(device)), target.to(device)) for data, target in val_loader)
    print(f'Validation Loss: {validation_loss / len(val_loader)}')

  # Save model
  if args.save:
    torch.save(model.state_dict(), f'weights/net{str(args.net)}.pth')

  model.eval()
  with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in test_dataset:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

if __name__ == '__main__':
  parser = argparse.ArgumentParser("Train a simple neural network on MNIST with PyTorch.")
  parser.add_argument("--net", type=int, default=1)
  parser.add_argument("--batch", type=int, default=16)
  parser.add_argument("--save", type=bool, default=False)
  args = parser.parse_args()
  main(args)
