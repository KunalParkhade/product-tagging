import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from base_model import GarmentClassifier
from torchmetrics import Accuracy

# Dataset and DataLoader
def load_data(batch_size=64):
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root='./data/raw', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data/raw', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Train Model
def train_model(epochs=2, learning_rate=0.001, batch_size=64):
    train_loader, _ = load_data(batch_size)
    model = GarmentClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_metric = Accuracy(task='multiclass', num_classes=10)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
    return model

if __name__ == "__main__":
    trained_model = train_model()
    torch.save(trained_model.state_dict(), "./models/garment_classifier.pth")