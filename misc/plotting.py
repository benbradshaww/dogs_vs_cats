import os
import random
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def plot_samples(samples, title: str = None):

    num_cols = 5
    num_rows = ((len(samples) - 1) // num_cols) + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

    axes = axes.flatten()

    for index, sample in enumerate(samples):
        image = mpimg.imread(sample)
        axes[index].imshow(image)
        axes[index].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, y=0.8)

    plt.tight_layout()
    plt.show()


def get_examples(num_samples: int):

    for animal in ["dogs", "cats"]:
        base_path = f"./data/training_set/training_set/{animal}/"
        files = os.listdir(base_path)
        images = [file for file in files if re.search(r"\.jpe?g$", file, re.IGNORECASE)]
        samples = random.sample(images, k=num_samples)
        samples = [base_path + sample for sample in samples]

        title = f"{animal} examples"
        plot_samples(samples, title=title)


def plot_loss(df: pd.DataFrame):

    epoch = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=epoch, y=train_loss, mode="lines", name="Train Loss"))

    fig.add_trace(go.Scatter(x=epoch, y=val_loss, mode="lines", name="Validation Loss"))

    fig.update_yaxes(range=[0, 0.8])

    fig.update_layout(title="Loss Graph", xaxis_title="Epoch", yaxis_title="Loss")

    fig.write_image("./plots/loss_plot.png")

    fig.show()


def plot_accuracy(df: pd.DataFrame):

    epoch = df["epoch"]
    train_acc = df["train_acc"]
    val_acc = df["val_acc"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=epoch, y=train_acc, mode="lines", name="Train Accuracy"))

    fig.add_trace(go.Scatter(x=epoch, y=val_acc, mode="lines", name="Validation Accuracy"))

    fig.update_yaxes(range=[0.5, 1])

    fig.update_layout(title="Accuracy Graph", xaxis_title="Epoch", yaxis_title="Accuracy")

    fig.write_image("./plots/accuracy_plot.png")

    fig.show()
