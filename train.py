import os
import torch
from resnet import device, criterion, optimizer, model


def train_model(model, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

    for images, labels, races in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/ {epochs}] Loss: {avg_loss: .4f}")


def test_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    all_races = []

    with torch.no_grad():
        for images, labels, races in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_races.extend(races)

    return all_preds, all_labels, all_races









