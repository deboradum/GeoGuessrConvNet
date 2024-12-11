import time
import torch
import optuna
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader

NUM_STATES = 51

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


def get_net():
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    net = torchvision.models.resnet50(weights=weights)
    net.fc = torch.nn.Linear(net.fc.in_features, NUM_STATES)
    torch.nn.init.xavier_uniform_(net.fc.weight)

    return net


def train(
    net, train_loader, test_loader, loss_fn, optimizer, epochs, device, filepath, patience, scheduler=None
):
    min_test_loss = 99999
    early_stopping_counter = 0
    for epoch in range(1, epochs+1):
        s = time.perf_counter()
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_acc += (torch.argmax(outputs, dim=1) == y).sum().item()

        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = net(X)
                loss = loss_fn(outputs, y)

                test_loss += loss.item() * X.size(0)
                test_acc += (torch.argmax(outputs, dim=1) == y).sum().item()

        time_taken = round(time.perf_counter()-s, 3)
        avg_train_loss, avg_train_acc = train_loss/len(train_loader.dataset), train_acc/len(train_loader.dataset)
        avg_test_loss, avg_test_acc = test_loss/len(test_loader.dataset), test_acc/len(test_loader.dataset)

        if scheduler:
            scheduler.step()

        print(
            f"Epoch: {epoch} | train loss: {avg_train_loss:.2f} | train acc: {avg_train_acc:.2f} | test loss: {avg_test_loss:.2f} |  test acc: {avg_test_acc:.2f} | Took {time_taken:.2f} seconds"
        )
        with open(filepath, "a+") as f:
            f.write(f"{epoch},{avg_train_loss},{avg_train_acc},{avg_test_loss},{avg_test_acc}\n")

        # Early stopping
        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                return min_test_loss

    return min_test_loss


def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_step_size = (
        trial.suggest_int("scheduler_step_size", 5, 20) if use_scheduler else 0
    )
    scheduler_gamma = (
        trial.suggest_uniform("scheduler_gamma", 0.1, 0.9) if use_scheduler else 0
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    train_loader, test_loader = get_dataloaders(root_dir="dataset/", batch_size=8)
    net = get_net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    lr, weight_decay = 1e-5, 5e-4
    params_1x = [
        param for name, param in net.named_parameters() if "fc" not in str(name)
    ]
    optimizer = torch.optim.Adam(
        [{"params": params_1x}, {"params": net.fc.parameters(), "lr": lr * 10}],
        lr=lr,
        weight_decay=weight_decay,
    )
    filepath = f"lr_{lr}_wd_{weight_decay}_scheduler_{use_scheduler}_stepsize_{scheduler_step_size}_gamma_{scheduler_gamma}.csv"
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
        epochs=25,
        device=device,
        filepath=filepath,
        patience=3,
        scheduler=scheduler,
    )

    return test_loss


if __name__ == "__main__":
    torch.manual_seed(101)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (test loss): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
