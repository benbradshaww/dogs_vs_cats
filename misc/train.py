import os
import time

import pandas as pd
import torch
from tqdm import tqdm


def accuracy(output, label):
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return round(correct / len(label), 3)


def precision_recall_f1(outputs, labels):
    output_pred = outputs.max(1)[1]
    true_positive = ((output_pred == 1) & (labels == 1)).sum().item()
    false_positive = ((output_pred == 1) & (labels == 0)).sum().item()
    false_negative = ((output_pred == 0) & (labels == 1)).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)

    f1_val = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1_val


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

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()

        outputs_tensor, labels_tensor = torch.empty(1, 2), torch.empty(1, 2)
        counter, running_loss = 0, 0
        for batch in tqdm(train_loader):
            features = batch[0]
            labels = batch[1]
            counter += 1

            optimizer.zero_grad()

            outputs = model(features)

            loss_train = criterion(outputs, labels)

            running_loss += loss_train
            outputs_tensor = torch.cat((outputs_tensor, outputs), dim=0)
            labels_tensor = torch.cat((labels_tensor, labels), dim=0)

            loss_train.backward()
            optimizer.step()

        train_loss = running_loss / counter
        train_acc = accuracy(outputs_tensor, labels_tensor)

        model.eval()
        outputs_tensor, labels_tensor = torch.empty(1, 2), torch.empty(1)
        counter, running_loss = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch[0]
                labels = batch[1]
                counter += 1

                outputs = model(features)

                outputs_tensor = torch.cat((outputs_tensor, outputs), dim=0)
                labels_tensor = torch.cat((labels_tensor, labels), dim=0)

                running_loss += criterion(outputs, labels)

        val_loss = running_loss / counter
        val_acc = accuracy(outputs_tensor, labels_tensor)

        # Changing step sizes
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Epoch {epoch}: Learning rate reduced from {current_lr} to {new_lr}")

        # Early stopping criterion
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
    outputs_tensor, labels_tensor = torch.empty(1, 2), torch.empty(1)
    counter, running_loss = 0, 0
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for batch in test_loader:
            labels = batch.y.squeeze().long()
            counter += 1

            outputs = model(batch)

            outputs_tensor = torch.cat((outputs_tensor, outputs), dim=0)
            labels_tensor = torch.cat((labels_tensor, labels), dim=0)

            running_loss += criterion(outputs, labels)

    test_loss = criterion(outputs, labels)
    test_acc = accuracy(outputs_tensor, labels_tensor)
    test_precision, test_recall, test_f1 = precision_recall_f1(outputs_tensor, labels_tensor)

    # f1_test = f1_metric(outputs, labels)
    print(
        "Test set results:",
        "loss= {:.4f}".format(test_loss),
        "accuracy= {:.4f}".format(test_acc),
        "recall= {:.4f}".format(test_recall),
        "precision= {:.4f}".format(test_precision),
        "f1= {:.4f}".format(test_f1),
    )
