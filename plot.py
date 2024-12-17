import matplotlib.pyplot as plt


def plot(paths, plot_train=True):
    # Initialize lists to hold metrics for each path
    epochs = []
    train_losses, test_losses = [], []
    train_acc_top1, train_acc_top3, train_acc_top5 = [], [], []
    test_acc_top1, test_acc_top3, test_acc_top5 = [], [], []

    # Parse data from each file
    for path in paths:
        epoch_vals, train_loss, test_loss = [], [], []
        acc_train1, acc_test1 = [], []
        acc_train3, acc_test3 = [], []
        acc_train5, acc_test5 = [], []

        with open(path, "r") as f:
            for line in f:
                (
                    epoch,
                    avg_train_loss,
                    avg_train_acc_top1,
                    avg_train_acc_top3,
                    avg_train_acc_top5,
                    avg_test_loss,
                    avg_test_acc_top1,
                    avg_test_acc_top3,
                    avg_test_acc_top5,
                ) = line.strip().split(",")

                epoch_vals.append(int(epoch))
                train_loss.append(float(avg_train_loss))
                test_loss.append(float(avg_test_loss))
                acc_train1.append(float(avg_train_acc_top1))
                acc_test1.append(float(avg_test_acc_top1))
                acc_train3.append(float(avg_train_acc_top3))
                acc_test3.append(float(avg_test_acc_top3))
                acc_train5.append(float(avg_train_acc_top5))
                acc_test5.append(float(avg_test_acc_top5))

        epochs.append(epoch_vals)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acc_top1.append(acc_train1)
        test_acc_top1.append(acc_test1)
        train_acc_top3.append(acc_train3)
        test_acc_top3.append(acc_test3)
        train_acc_top5.append(acc_train5)
        test_acc_top5.append(acc_test5)

    metrics = [
        (train_losses, test_losses, "Loss", "loss_plot.png"),
        (train_acc_top1, test_acc_top1, "Top-1 accuracy", "accuracy_top1_plot.png"),
        (train_acc_top3, test_acc_top3, "Top-3 accuracy", "accuracy_top3_plot.png"),
        (train_acc_top5, test_acc_top5, "Top-5 accuracy", "accuracy_top5_plot.png"),
    ]

    # Define colors for each path
    colors = plt.cm.tab10(range(len(paths)))

    # Generate and save each plot
    for train, test, title, filename in metrics:
        plt.figure(figsize=(10, 5))
        for j, (epoch, train_vals, test_vals) in enumerate(zip(epochs, train, test)):
            label = paths[j].split("_")[0].split("/")[-1]
            if plot_train:
                plt.plot(
                    epoch,
                    train_vals,
                    label=f"Train {label}",
                    color=colors[j],
                )
            plt.plot(
                epoch,
                test_vals,
                label=f"{label}",
                color=colors[j],
                linestyle="--",
            )
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


paths = [
    "results/resnet34_results/resnet34_lr_5.271243178881065e-05_wd_1.9967021251960164e-06_stepsize_5_gamma_0.8145093310551305_dropout_0.4.csv",
    "results/resnet50_results/resnet50_lr_5.271243178881065e-05_wd_1.9967021251960164e-06_stepsize_5_gamma_0.8145093310551305_dropout_0.4.csv",
    "results/resnet101_results/resnet101_lr_5.271243178881065e-05_wd_1.9967021251960164e-06_stepsize_5_gamma_0.8145093310551305_dropout_0.4.csv",
    "results/resnet152_results/resnet152_lr_5.271243178881065e-05_wd_1.9967021251960164e-06_stepsize_5_gamma_0.8145093310551305_dropout_0.4.csv",
]
plot(paths, plot_train=False)
