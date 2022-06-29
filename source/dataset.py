from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import config
import matplotlib.pyplot as plt

Transform = transforms.Compose([
    transforms.Resize((config.input_size, config.input_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='./dataset/', train=True, transform=Transform, download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

test_dataset = datasets.CIFAR100(root='./dataset/', train=False, transform=Transform, download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

if __name__ == '__main__':
    datatiter = iter(train_loader)
    data = datatiter.next()
    Image, label = data
    print(Image[0].size())
    Img_np = Image[0].numpy()
    plt.imshow(Image[0].permute(1,2,0))
    plt.show()
    print(label)