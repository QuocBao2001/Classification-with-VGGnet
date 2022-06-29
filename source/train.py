import torch
import torch.nn as nn
from torch import optim
from VGG16_model import VGG16_net
from dataset import train_loader, test_loader
import config
from tqdm import tqdm
from time import sleep

# get computer device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# set model to device
model = VGG16_net(image_size=config.input_size, in_channels=config.img_channels,
                  numclasses=config.num_classes).to(device)

# create cross-entropy for classification
criterion = nn.CrossEntropyLoss()
# optim by Adam
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

if config.load_model:
    load_checkpoint(config.checkpoint_file, model, optimizer, config.learning_rate)

# training
for epoch in range(config.num_epochs):
    with tqdm(train_loader, unit='batch') as tepoch:
        for data, targets in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            predictions = model(data)
            loss = criterion(predictions, targets)

            # check current accuracy
            correct = 0
            for i in range(config.batch_size):
                if predictions[i].argmax() == targets[i]:
                    correct += 1
            accuracy = correct / config.batch_size

            # backward
            optimizer.zero_grad()
            loss.backward()

            # adam step
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100.*accuracy)
    if config.save_checkpoint:
        save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

    # check accuracy on test
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(100)]
        n_class_samples = [0 for i in range(100)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(config.batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(100):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {i}: {acc} %')
