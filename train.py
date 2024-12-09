import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader


def get_dataloaders(root_dir="dataset/", batch_size=16):
    dataset = datasets.ImageFolder(
        root=root_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader


def train():
    return


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(root_dir="dataset/", batch_size=16)

    for i_batch, (X, y) in enumerate(train_loader):
        pass
