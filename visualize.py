import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import get_dataloaders, get_net
from sklearn.metrics import classification_report


def create_prediction_figure(image, guess, ground_truth, output_path):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    image = (image - image.min()) / (image.max() - image.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
    plt.axis("off")
    plt.title(f"Prediction: {guess}\nGround Truth: {ground_truth}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# Dataloader should use a batch size of 60 (the number of test images per class)
def visualize_predictions(net, dataloader, output_dir):
    classes = dataloader.dataset.classes
    for inputs, labels in dataloader:
        logits = net(inputs)
        predictions = torch.argmax(logits, dim=1).tolist()
        ground_truth = labels.tolist()

        inputs.tolist()

        for i, (p, gt) in enumerate(zip(predictions, ground_truth)):
            guess_class = classes[p]
            gt_class = classes[gt]
            img_number = i % 61  # 60 images per class. Assumes dataset is not shuffled.
            output_path = f"{output_dir}/{gt_class}/{img_number}.png"
            create_prediction_figure(inputs[i], guess_class, gt_class, output_path)


def get_accuracies(net, dataloader, output_path):
    all_predictions = []
    all_ground_truth = []
    for inputs, labels in dataloader:
        logits = net(inputs)
        predictions = torch.argmax(logits, dim=1).tolist()
        ground_truth = labels.tolist()

        all_predictions.extend(predictions)
        all_ground_truth.extend(ground_truth)
    classes = dataloader.dataset.classes

    report_dict = classification_report(
        all_ground_truth, all_predictions, target_names=classes, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    pretrained_path = "resnet101_lr_5.271243178881065e-05_wd_1.9967021251960164e-06_stepsize_5_gamma_0.8145093310551305_dropout_0.4.pth"
    resnet = pretrained_path.split("_")[0]

    _, test_loader, num_classes = get_dataloaders(
        root_dir="countryDataset/", batch_size=60
    )

    net = get_net(num_classes, resnet=resnet)
    net.load_state_dict(torch.load(pretrained_path, weights_only=True))
    net.eval()

    visualize_predictions(net, test_loader, f"{resnet}_results")
    get_accuracies(net, test_loader, f"{resnet}_results.csv")
