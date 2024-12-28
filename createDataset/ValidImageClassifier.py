import time
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_valid(path):
    return path.endswith(".png")


def get_dataloaders(root_dir="dataset/", batch_size=16):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_dataset = datasets.ImageFolder(
        root=f"{root_dir}/train/",
        transform=transforms.Compose(
            [
                transforms.RandomCrop(
                    (896, 896),
                ),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        ),
        is_valid_file=is_valid,
    )
    test_dataset = datasets.ImageFolder(
        root=f"{root_dir}/test/",
        transform=transforms.Compose(
            [
                transforms.RandomCrop(
                    (448, 448),
                ),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        ),
        is_valid_file=is_valid,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, test_loader, len(train_dataset.classes)


def get_net():
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    resnet_model = torchvision.models.resnet18
    net = resnet_model(weights=weights)

    net.fc = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(net.fc.in_features, 1),

    )
    torch.nn.init.xavier_uniform_(net.fc[1].weight)

    return net

def get_acc(logits, targets):
    preds = torch.where(torch.sigmoid(logits)>0.5, 1, 0).squeeze(1)
    corr = (preds==targets).float().sum()
    return corr

    acc = corr / preds.numel()

    return acc


def train(
    net,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs,
    device,
    filepath,
    patience,
):

    min_test_loss = 99999
    early_stopping_counter = 0
    for epoch in range(1, epochs + 1):
        s = time.perf_counter()
        net.train()
        train_loss = 0.0

        train_acc_top1 = 0.0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            loss = loss_fn(outputs, y.view(-1, 1).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            top1_acc = get_acc(outputs, y)
            train_acc_top1 += top1_acc.item()

        net.eval()
        test_loss = 0.0
        test_acc_top1 = 0.0
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)
                outputs = net(X)
                loss = loss_fn(outputs, y.view(-1, 1).float())

                test_loss += loss.item() * X.size(0)
                top1_acc = get_acc(outputs, y)
                test_acc_top1 += top1_acc.item()

        time_taken = round(time.perf_counter() - s, 3)
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc_top1 / len(train_loader.dataset)
        avg_test_loss = test_loss / len(test_loader.dataset)
        avg_test_acc = test_acc_top1 / len(test_loader.dataset)

        print(
            f"Epoch: {epoch} | train loss: {avg_train_loss:.2f} | train acc: {avg_train_acc:.2f} | test loss: {avg_test_loss:.2f} | test acc {avg_test_acc:.2f} | Took {time_taken:.2f} seconds"
        )
        with open(filepath, "a+") as f:
            f.write(
                f"{epoch},{avg_train_loss},{avg_test_loss}\n"
            )

        # Early stopping
        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            early_stopping_counter = 0
            best_state_dict = net.state_dict().copy()
            final_path = filepath.replace(".csv", ".pth")
            torch.save(best_state_dict, final_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                final_path = filepath.replace(".csv", ".pth")
                best_model = torchvision.models.resnet18()
                best_state_dict = torch.load(
                    final_path, map_location=torch.device("mps")
                )
                best_model.load_state_dict(best_state_dict)
                best_model.eval()
                test_loss = 0.0
                test_acc_top1 = 0.0
                with torch.no_grad():
                    for i, (X, y) in enumerate(test_loader):
                        X, y = X.to(device), y.to(device)
                        outputs = best_model(X)
                        loss = loss_fn(outputs, y.view(-1, 1).float())

                        test_loss += loss.item() * X.size(0)
                        top1_acc = get_acc(outputs, y)
                        test_acc_top1 += top1_acc.item()
                avg_test_loss = test_loss / len(test_loader.dataset)
                avg_test_acc = test_acc_top1 / len(test_loader.dataset)
                print("Final loss and acc:", avg_test_loss, avg_test_acc)
                return min_test_loss


    final_path = filepath.replace(".csv", ".pth")
    torch.save(net.state_dict(), final_path)

    best_model = torch.load(
        final_path, map_location=torch.device("mps"), weights_only=True
    )
    best_model.eval()
    test_loss = 0.0
    test_acc_top1 = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            outputs = best_model(X)
            loss = loss_fn(outputs, y.view(-1, 1).float())

            test_loss += loss.item() * X.size(0)
            top1_acc = get_acc(outputs, y)
            test_acc_top1 += top1_acc.item()
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_acc = test_acc_top1 / len(test_loader.dataset)
    print("Final loss and acc:", avg_test_loss, avg_test_acc)

    return min_test_loss


if __name__ == "__main__":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    net = get_net().to(device)
    train_loader, test_loader, num_classes = get_dataloaders(
        root_dir="validImageClassifierDataset/", batch_size=64
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    params_1x = [
        param for name, param in net.named_parameters() if "fc" not in str(name)
    ]
    lr = 5.271243178881065e-5
    weight_decay = 1.9967021251960164e-6
    optimizer = torch.optim.Adam(
        [{"params": params_1x}, {"params": net.fc[1].parameters(), "lr": lr * 10}],
        lr=lr,
        weight_decay=weight_decay,
    )

    train(
        net,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        5,
        device,
        "model.csv",
        3,
    )
