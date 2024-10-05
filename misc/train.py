import os
import time

import pandas as pd
import torch
from tqdm import tqdm


def accuracy(output, label):
    preds = output.argmax(dim=1)
    label = label.argmax(dim=1)
    correct = preds.eq(label).sum().item()
    return round(correct / len(label), 3)


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    model_path,
    epochs: int,
    seed=42,
    patience: int = 15,
    threshold: float = 1e-4,
):

    best_loss = float("inf")
    start_time = time.time()
    criterion = torch.nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):

        epoch_start_time = time.time()
        model.train()

        counter = 0
        running_loss = 0
        total_samples = 0
        running_corrects = 0

        for batch in tqdm(train_loader):
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            counter += 1

            optimizer.zero_grad()

            outputs = model(images)

            loss_train = criterion(outputs, labels)

            running_loss += loss_train.item()

            preds = outputs.argmax(1)
            labels_1_dim = labels.argmax(1)

            running_corrects += torch.sum(preds == labels_1_dim).item()
            total_samples += labels.size(0)

            loss_train.backward()
            optimizer.step()

        train_loss = running_loss / counter
        train_acc = running_corrects / total_samples

        # Validation

        model.eval()
        counter = 0
        running_loss = 0
        total_samples = 0
        running_corrects = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch[0], batch[1]
                images, labels = images.to(device), labels.to(device)
                counter += 1

                outputs = model(images)

                running_loss += criterion(outputs, labels).item()

                preds = outputs.argmax(1)
                labels_1_dim = labels.argmax(1)

                running_corrects += torch.sum(preds == labels_1_dim).item()
                total_samples += labels.size(0)

        val_loss = running_loss / counter
        val_acc = running_corrects / total_samples

        # Changing step sizes
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Epoch {epoch}: Learning rate reduced from {current_lr} to {new_lr}")

        # Early stopping criterion
        val_loss = val_loss
        if val_loss < best_loss - threshold:
            best_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

        new_row = pd.DataFrame(
            {
                "epoch": epoch + 1,
                "train_loss": [train_loss],
                "train_acc": [train_acc],
                "val_loss": [val_loss],
                "val_acc": [val_acc],
            }
        )

        history_path = "./history/history.csv"
        os.makedirs("./history", exist_ok=True)
        if epoch == 0 and os.path.isfile(history_path):
            os.remove(history_path)
            new_row.to_csv(history_path, mode="a", header=True, index=False)
        else:
            new_row.to_csv(history_path, mode="a", header=False, index=False)

        if ((epoch + 1) % 1) == 0:
            print(
                "Epoch: {:04d}".format(epoch + 1),
                "loss_train: {:.4f}".format(train_loss),
                "acc_train: {:.4f}".format(train_acc),
                "loss_val: {:.4f}".format(val_loss),
                "acc_val: {:.4f}".format(val_acc),
                "time: {:.4f}s".format(time.time() - epoch_start_time),
            )

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))


def test_model(model, test_loader):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCELoss()

    counter = 0
    running_loss = 0
    total_samples = 0
    running_corrects = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            counter += 1

            outputs = model(images)

            preds = outputs.argmax(1)
            label_1_dim = labels.argmax(1)

            running_corrects += torch.sum(preds == label_1_dim).item()
            total_samples += labels.size(0)

            running_loss += criterion(outputs, labels)

    test_loss = running_loss / counter
    test_acc = running_corrects / total_samples

    print("Test set results:", "loss= {:.4f}".format(test_loss), "accuracy= {:.4f}".format(test_acc))
