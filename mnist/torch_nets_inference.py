import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from torch_nets import Net3

model = Net3()
model.load_state_dict(torch.load('models/net3.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def show_mnist_predictions(dataset, model, num_images):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    
    for i, ax in enumerate(axes):
        image, true_label = dataset[i]
        
        image_batch = image.unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_batch)
            _, predicted_label = torch.max(output, 1)
            predicted_label = predicted_label.item()
        
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f'Label: {true_label}\nPred: {predicted_label}')
        ax.axis('off')
    
    plt.show()

show_mnist_predictions(test_dataset, model, num_images=6)
