import matplotlib.pyplot as plt


def plot(paths, plot_train=True):
    # Initialize lists to hold metrics for each path
    epochs = []
    train_losses = []
    test_losses = []
    train_acc_top1 = []
    test_acc_top1 = []
    train_acc_top3 = []
    test_acc_top3 = []
    train_acc_top5 = []
    test_acc_top5 = []

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

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    metrics = [
        (train_losses, test_losses, "Loss"),
        (train_acc_top1, test_acc_top1, "Accuracy Top-1"),
        (train_acc_top3, test_acc_top3, "Accuracy Top-3"),
        (train_acc_top5, test_acc_top5, "Accuracy Top-5"),
    ]

    # Define colors for each path
    colors = plt.cm.tab10(range(len(paths)))


    # Plot each metric
    for i, (train, test, title) in enumerate(metrics):
        ax = axes[i]
        for j, (epoch, train_vals, test_vals) in enumerate(zip(epochs, train, test)):
            label = paths[j]
            if plot_train:
                ax.plot(
                    epoch,
                    train_vals,
                    label=f"Train - {label}",
                    color=colors[j],
                )
            ax.plot(
                epoch,
                test_vals,
                label=f"Test - {label}",
                color=colors[j],
                linestyle="--",
            )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend()
        ax.grid()

    axes[-1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.show()


paths = [
    "lr_3.551149703221821e-05_wd_3.50789550172109e-05_scheduler_False_stepsize_0_gamma_0.csv",
    "lr_3.64948994387593e-05_wd_2.3583623069232676e-06_scheduler_False_stepsize_0_gamma_0.csv",
    "lr_4.6665284807168925e-05_wd_5.411810799060977e-06_scheduler_False_stepsize_0_gamma_0.csv",
    "lr_4.871185119221744e-06_wd_1.1482078272972141e-06_scheduler_True_stepsize_19_gamma_0.8847210406663705.csv",
    "lr_5.271243178881065e-05_wd_1.0584920651159921e-05_scheduler_True_stepsize_11_gamma_0.14202500516101513.csv",
]
plot(paths, plot_train=False)
