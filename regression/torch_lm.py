import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

class LinearRegression(nn.Module):
  def __init__(self):
    super(LinearRegression, self).__init__()
    self.linear = nn.Linear(1, 1)

  def forward(self, x):
    return self.linear(x)

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.layer1 = nn.Linear(in_features=1, out_features=10)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(in_features=10, out_features=1)

  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.layer2(x)
    return x

# model = LinearRegression()
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x_limits = [0, 9]
y_limits = [0, 9]

epochs = 200
for epoch in range(epochs):
  optimizer.zero_grad()
  outputs = model(x)
  loss = criterion(outputs, y)
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 1 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, model(x).detach().numpy(), label='Fitted line')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.legend()
    plt.draw()  # Use plt.draw() to render the plot
    plt.pause(0.05)  # Use plt.pause() to ensure the plot gets rendered

    # print("Press any key to continue.")
    # plt.waitforbuttonpress()  # Waits for a button press to continue
    plt.clf()  # Clears the current figure