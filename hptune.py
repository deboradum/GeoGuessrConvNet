import time
import torch
import optuna
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


def is_valid(path):
    return path.endswith(".png")

def get_dataloaders(root_dir="dataset/", batch_size=16):
    train_dataset = datasets.ImageFolder(
        root=f"{root_dir}/train/",
        transform=transforms.Compose(
            [
                transforms.RandomCrop(
                    (448, 448),
                ),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
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

def get_resnet(num_classes, resnet, dropout_rate):
    if resnet == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        resnet_model = torchvision.models.resnet18
    elif resnet == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        resnet_model = torchvision.models.resnet34
    elif resnet == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        resnet_model = torchvision.models.resnet50
    elif resnet == "resnet101":
        weights = torchvision.models.ResNet101_Weights.DEFAULT
        resnet_model = torchvision.models.resnet101
    elif resnet == "resnet152":
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        resnet_model = torchvision.models.resnet152
    else:
        print(f"{resnet} not supported")
        exit(0)

    net = resnet_model(weights=weights)

    net.fc = torch.nn.Sequential(
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(net.fc.in_features, num_classes),
    )
    torch.nn.init.xavier_uniform_(net.fc[1].weight)

    return net

def get_vit(num_classes, net_name, dropout_rate):
    if net_name == "vit_b_16":
        weights = torchvision.models.vision_transformer.ViT_B_16_Weights.DEFAULT
        vit_model = torchvision.models.vision_transformer.vit_b_16
    elif net_name == "vit_b_32":
        weights = torchvision.models.vision_transformer.ViT_B_32_Weights.DEFAULT
        vit_model = torchvision.models.vision_transformer.vit_b_32
    elif net_name == "vit_l_16":
        weights = torchvision.models.vision_transformer.ViT_L_16_Weights.DEFAULT
        vit_model = torchvision.models.vision_transformer.vit_l_16
    else:
        print(f"{net_name} not supported")
        exit(0)

    net = vit_model(weights=weights)

    net.heads.head = torch.nn.Sequential(
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(net.heads.head.in_features, num_classes),
    )
    torch.nn.init.xavier_uniform_(net.heads.head[1].weight)

    return net


def get_efficientnet(num_classes, net_name, dropout_rate):
    if net_name == "efficientnet_b0":
        weights = torchvision.models.efficientnet.EfficientNet_B0_Weights.DEFAULT
        efficientnet_model = torchvision.models.efficientnet_b0
    elif net_name == "efficientnet_b1":
        weights = torchvision.models.efficientnet.EfficientNet_B1_Weights.DEFAULT
        efficientnet_model = torchvision.models.efficientnet_b1
    elif net_name == "efficientnet_b2":
        weights = torchvision.models.efficientnet.EfficientNet_B2_Weights.DEFAULT
        efficientnet_model = torchvision.models.efficientnet_b2
    else:
        print(f"{net_name} not supported")
        exit(0)

    net = efficientnet_model(weights=weights)

    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, num_classes)
    net.classifier.add_module("dropout", torch.nn.Dropout(dropout_rate))
    torch.nn.init.xavier_uniform_(net.classifier[1].weight)

    return net


def get_net(num_classes, net_name="resnet50", dropout_rate=0.5):
    if "resnet" in net_name:
        return get_resnet(num_classes, net_name, dropout_rate)
    elif "vit" in net_name:
        return get_vit(num_classes, net_name, dropout_rate)
    elif "efficientnet" in net_name:
        return get_efficientnet(num_classes, net_name, dropout_rate)
    else:
        print(f"{net_name} not supported")
        exit(0)


def correct_predictions(output, target, topk=(1,)):
    maxk = max(topk)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)

    pred = pred.t()
    correct = pred.eq(
        target.view(1, -1).expand_as(pred)
    )

    ret = []
    for k in topk:
        # Number of correct outputs is normalized later to get accuracy.
        correct_k = (
            correct[:k].reshape(-1).float().sum(0, keepdim=True)
        )
        ret.append(correct_k)
    return ret


def train(
    net, train_loader, test_loader, loss_fn, optimizer, epochs, device, run_name, patience, writer, scheduler=None
):
    min_test_loss = 99999
    early_stopping_counter = 0
    for epoch in range(1, epochs+1):
        s = time.perf_counter()
        net.train()

        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1_acc, top3_acc, top5_acc = correct_predictions(outputs, y, (1, 3, 5))

            time_step = (epoch - 1) * len(train_loader.dataset) + (i + 1) * len(X)
            writer.add_scalar("Loss/train", loss.item(), time_step)
            writer.add_scalar("Accuracy_t1/train", top1_acc, time_step)
            writer.add_scalar("Accuracy_t3/train", top3_acc, time_step)
            writer.add_scalar("Accuracy_t5/train", top5_acc, time_step)

        net.eval()
        test_loss = 0.0
        test_acc_top1 = 0.0
        test_acc_top3 = 0.0
        test_acc_top5 = 0.0
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)
                outputs = net(X)
                loss = loss_fn(outputs, y)

                test_loss += loss.item() * X.size(0)
                top1_acc, top3_acc, top5_acc = correct_predictions(outputs, y, (1, 3, 5))
                test_acc_top1 += top1_acc.item()
                test_acc_top3 += top3_acc.item()
                test_acc_top5 += top5_acc.item()


        time_taken = round(time.perf_counter()-s, 3)

        avg_test_loss  = test_loss/len(test_loader.dataset)
        avg_test_acc_top1, avg_test_acc_top3, avg_test_acc_top5 = (
            test_acc_top1 / len(test_loader.dataset),
            test_acc_top3 / len(test_loader.dataset),
            test_acc_top5 / len(test_loader.dataset),
        )

        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy_t1/test", avg_test_acc_top1, epoch)
        writer.add_scalar("Accuracy_t3/test", avg_test_acc_top3, epoch)
        writer.add_scalar("Accuracy_t5/test", avg_test_acc_top5, epoch)

        if scheduler:
            scheduler.step()

        print(
            f"Epoch: {epoch} | test loss: {avg_test_loss:.2f} |  top 1 test acc: {avg_test_acc_top1:.2f} |  top 3 test acc: {avg_test_acc_top3:.2f} |  top 5 test acc: {avg_test_acc_top5:.2f} | Took {time_taken:.2f} seconds"
        )

        # Early stopping
        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                torch.save(net.state_dict(), f"{run_name}.pth")
                return min_test_loss

    torch.save(net.state_dict(), f"{run_name}.pth")
    return min_test_loss


def objective(trial):
    net_name = "resnet50"
    lr = trial.suggest_float("lr", 0.0001, 0.00001, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.45, 0.65)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_step_size = (
        trial.suggest_int("scheduler_step_size", 5, 20) if use_scheduler else 0
    )
    scheduler_gamma = (
        trial.suggest_float("scheduler_gamma", 0.1, 0.9) if use_scheduler else 0
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    train_loader, test_loader, num_classes = get_dataloaders(
        root_dir="countryDataset/", batch_size=128
    )
    num_classes = 85
    net = get_net(num_classes, net_name=net_name, dropout_rate=dropout_rate).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    if "resnet" in net_name:
        params_1x = [
            param for name, param in net.named_parameters() if "fc" not in str(name)
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": params_1x},
                {"params": net.fc.parameters(), "lr": lr * 10},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
    elif "vit" in net_name:
        params_1x = [
            param
            for name, param in net.named_parameters()
            if "head" not in str(name)
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": params_1x},
                {"params": net.heads.head.parameters(), "lr": lr * 10},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
    elif "efficientnet" in net_name:
        params_1x = [
            param
            for name, param in net.named_parameters()
            if "classifier" not in str(name)
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": params_1x},
                {"params": net.classifier.parameters(), "lr": lr * 10},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )

    run_name = f"{net_name}_lr_{lr}_wd_{weight_decay}_scheduler_{use_scheduler}_stepsize_{scheduler_step_size}_gamma_{scheduler_gamma}_dropout_{dropout_rate}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )

    test_loss = train(
        net=net,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        epochs=5,
        device=device,
        run_name=run_name,
        patience=3,
        scheduler=scheduler,
        writer=writer,
    )

    return test_loss


if __name__ == "__main__":
    torch.manual_seed(4214)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (test loss): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
