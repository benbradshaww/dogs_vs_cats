import os
import random
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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
