import os
import umap
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
from train import get_dataloaders, get_net


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
def get_accuracies(net, dataloader, output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    classes = dataloader.dataset.classes
    all_predictions = []
    all_ground_truth = []
    for inputs, labels in dataloader:
        logits = net(inputs)
        predictions = torch.argmax(logits, dim=1).tolist()
        ground_truth = labels.tolist()

        all_predictions.extend(predictions)
        all_ground_truth.extend(ground_truth)

        # Visualize prediction and ground truth
        inputs.tolist()
        for i, (p, gt) in enumerate(zip(predictions, ground_truth)):
            guess_class = classes[p]
            gt_class = classes[gt]
            img_number = i % 61  # 60 images per class. Assumes dataset is not shuffled.
            output_path = f"{output_dir}/visualizations/{gt_class}/{img_number}.png"
            create_prediction_figure(inputs[i], guess_class, gt_class, output_path)
    classes = dataloader.dataset.classes

    report_dict = classification_report(
        all_ground_truth, all_predictions, target_names=classes, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"{output_dir}/accuracies.csv", index=True)

    cm = confusion_matrix(all_ground_truth, all_predictions)

    return cm, classes


def plot_tsne_umap(cm, classes, base_output_path):
    # Normalize the confusion matrix by row to get similarity scores
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_embeddings = tsne.fit_transform(cm_normalized)

    # UMAP
    umap_model = umap.UMAP(n_neighbors=15, random_state=42)
    umap_embeddings = umap_model.fit_transform(cm_normalized)

    # Cosine similarity t-SNE
    cosine_sim = cosine_similarity(cm_normalized)  # Compute cosine similarity
    tsne_cosine = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    cosine_embeddings = tsne_cosine.fit_transform(cosine_sim)

    # Cosine Similarity UMAP
    umap_cosine = umap.UMAP(n_neighbors=15, random_state=42)
    cosine_embeddings_umap = umap_cosine.fit_transform(cosine_sim)

    #  t-SNE
    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=np.arange(len(classes)),
        cmap="tab20",
        edgecolor="k",
        s=100,
    )
    plt.title("t-SNE class confusion visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    for i, label in enumerate(classes):
        plt.annotate(label, (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), fontsize=8)
    tsne_output_path = f"{base_output_path}_TSNE.png"
    plt.savefig(tsne_output_path, bbox_inches="tight")
    plt.close()

    # UMAP
    plt.figure(figsize=(10, 8))
    plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=np.arange(len(classes)),
        cmap="tab20",
        edgecolor="k",
        s=100,
    )
    plt.title("UMAP class confusion visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    for i, label in enumerate(classes):
        plt.annotate(label, (umap_embeddings[i, 0], umap_embeddings[i, 1]), fontsize=8)
    umap_output_path = f"{base_output_path}_UMAP.png"
    plt.savefig(umap_output_path, bbox_inches="tight")
    plt.close()

    # Cosine Similarity t-SNE Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        cosine_embeddings[:, 0],
        cosine_embeddings[:, 1],
        c=np.arange(len(classes)),
        cmap="tab20",
        edgecolor="k",
        s=100,
    )
    plt.title("t-SNE cosine similarity visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    for i, label in enumerate(classes):
        plt.annotate(
            label, (cosine_embeddings[i, 0], cosine_embeddings[i, 1]), fontsize=8
        )
    cosine_output_path = f"{base_output_path}_Cosine_TSNE.png"
    plt.savefig(cosine_output_path, bbox_inches="tight")
    plt.close()

    # Cosine Similarity UMAP Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        cosine_embeddings_umap[:, 0],
        cosine_embeddings_umap[:, 1],
        c=np.arange(len(classes)),
        cmap="tab20",
        edgecolor="k",
        s=100,
    )
    plt.title("UMAP cosine similarity visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    for i, label in enumerate(classes):
        plt.annotate(
            label,
            (cosine_embeddings_umap[i, 0], cosine_embeddings_umap[i, 1]),
            fontsize=8,
        )
    cosine_umap_output_path = f"{base_output_path}_Cosine_UMAP.png"
    plt.savefig(cosine_umap_output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    pretrained_path = "resnet101_lr_5.271243178881065e-05_wd_1.9967021251960164e-06_stepsize_5_gamma_0.8145093310551305_dropout_0.4.pth"
    resnet = pretrained_path.split("_")[0]

    _, test_loader, num_classes = get_dataloaders(
        root_dir="countryDataset/", batch_size=60
    )

    net = get_net(num_classes, resnet=resnet)
    net.load_state_dict(torch.load(pretrained_path, weights_only=True))
    net.eval()

    # Get accuracies in csv format and visualize predictions.
    cm, classes = get_accuracies(net, test_loader, f"{resnet}_results")

    # Plot accuracies with tsne and umap
    plot_tsne_umap(cm, classes, f"{resnet}_results/{resnet}_class_confusions")
