from typing import Any, Tuple
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from model import DigitClassifierModel
import numpy as np
import matplotlib.pyplot as plt


loss_fn = CrossEntropyLoss()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


def show5(img_loader):
    dataiter = iter(img_loader)
    
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(int(labels[i].detach()))
    
        image = images[i].numpy()
        plt.imshow(image.T.squeeze().T)
        plt.show()

def get_data_loader() -> Tuple[Any, Any]:
    training_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
    training_loader = DataLoader(dataset=training_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)
    return training_loader, test_loader


def training(training_loader: DataLoader, model: DigitClassifierModel, loss_fn: CrossEntropyLoss, optimizer: torch.optim.Adam):
    model.train()
    loss = np.float32('-inf')
    for idx, (images, labels) in enumerate(training_loader):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    return loss
    
def testing(testing_loader: DataLoader, model: DigitClassifierModel, loss_fn: CrossEntropyLoss):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testing_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = test_loss / len(testing_loader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    testing_loader, training_loader = get_data_loader()
    epochs = 10

    net = DigitClassifierModel()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    training_loss_history = []
    validation_loss_history = []
    validation_accuracy_history = []
    for epoch in range(epochs):
        training_loss = training(training_loader, net, loss_fn, optimizer)
        val_loss, val_acc = testing(testing_loader, net, loss_fn)

        training_loss_history.append(training_loss.item())
        validation_loss_history.append(val_loss)
        validation_accuracy_history.append(val_acc)
        print(f"Accuracy {val_acc}")

    plt.plot(training_loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss.png", dpi=300, bbox_inches='tight')
    plt.cla()

    plt.plot(validation_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("validation_loss.png", dpi=300, bbox_inches='tight')
    plt.cla()


    plt.plot(validation_accuracy_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("training_accuracy.png", dpi=300, bbox_inches='tight')

    torch.save(net.state_dict(), "digit_classifier.pth")


        

    