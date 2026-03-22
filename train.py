import os
import torch
from resnet import device, criterion, optimiser, model


def train_model(model, train_loader, val_loader, epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
    
        total_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/ {epochs}] Loss: {avg_loss: .4f}")




