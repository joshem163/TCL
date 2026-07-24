import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def run_tsne(embeddings, seed, perplexity):
    embeddings = embeddings.numpy()

    embeddings = StandardScaler().fit_transform(
        embeddings
    )

    # Perplexity must be smaller than the number of samples.
    perplexity = min(
        perplexity,
        len(embeddings) - 1
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed
    )

    return tsne.fit_transform(embeddings)


def plot_embedding(
    embeddings,
    labels,
    title,
    output_file,
    seed,
    perplexity
):
    coordinates = run_tsne(
        embeddings,
        seed,
        perplexity
    )

    labels = labels.numpy()

    plt.figure(figsize=(7, 6))

    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        s=40,
        alpha=0.8
    )

    plt.colorbar(
        scatter,
        label="Class"
    )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding_file",
        type=str,
        # required=True,
        default='saved_embeddings/bzr_fold_1_embeddings.pt'
    )

    parser.add_argument(
        "--epoch",
        type=int,
        # required=True,
        default=25
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="tsne_figures"
    )

    parser.add_argument(
        "--perplexity",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    embedding_history = torch.load(
        args.embedding_file,
        map_location="cpu"
    )

    selected_epoch = None

    for epoch_data in embedding_history:
        if epoch_data["epoch"] == args.epoch:
            selected_epoch = epoch_data
            break

    if selected_epoch is None:
        available_epochs = [
            item["epoch"]
            for item in embedding_history
        ]

        raise ValueError(
            f"Epoch {args.epoch} was not found. "
            f"Available epochs: {available_epochs}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    labels = selected_epoch["labels"]

    plot_embedding(
        embeddings=selected_epoch["gin"],
        labels=labels,
        title=f"GIN Embedding — Epoch {args.epoch}",
        output_file=os.path.join(
            args.output_dir,
            f"gin_epoch_{args.epoch}.png"
        ),
        seed=args.seed,
        perplexity=args.perplexity
    )

    plot_embedding(
        embeddings=selected_epoch["mlp"],
        labels=labels,
        title=f"MLP Embedding — Epoch {args.epoch}",
        output_file=os.path.join(
            args.output_dir,
            f"mlp_epoch_{args.epoch}.png"
        ),
        seed=args.seed,
        perplexity=args.perplexity
    )

    plot_embedding(
        embeddings=selected_epoch["final"],
        labels=labels,
        title=f"Final Embedding — Epoch {args.epoch}",
        output_file=os.path.join(
            args.output_dir,
            f"final_epoch_{args.epoch}.png"
        ),
        seed=args.seed,
        perplexity=args.perplexity
    )


if __name__ == "__main__":
    main()