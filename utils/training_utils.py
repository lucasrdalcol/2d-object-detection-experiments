from tqdm import tqdm
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.yolov1_config as cfg


# Train function
def train_epoch(train_dataloader, model, optimizer, loss_fn):
    train_dataloader_loop = tqdm(train_dataloader)
    model.train()  # set model to training mode
    losses = []

    # Iterate over the training data
    for batch_idx, (inputs_x, labels_y) in enumerate(train_dataloader_loop):
        inputs_x, labels_y = inputs_x.to(cfg.DEVICE), labels_y.to(
            cfg.DEVICE
        )  # Move data to device (GPU if available)

        # forward pass: Feed inputs to the model and compute loss.
        output = model(inputs_x)  # Feed inputs to the model to get predictions
        loss = loss_fn(output, labels_y)  # calculate loss
        losses.append(loss.item())

        # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()  # zero out the gradients from the previous step
        loss.backward()  # backpropagate the loss
        optimizer.step()  # perform a single optimization step (model parameter update)

        train_dataloader_loop.set_postfix(loss=loss.item())  # update progress bar

    # Compute mean loss over all batches of the training data
    mean_loss = sum(losses) / len(losses)
    print(f"Train mean loss: {mean_loss}")


def validate_epoch(val_dataloader, model, loss_fn):
    val_dataloader_loop = tqdm(val_dataloader)
    model.eval()  # set model to evaluation mode
    losses = []

    # Iterate over the validation data
    with torch.no_grad():  # disable gradient calculation
        for batch_idx, (inputs_x, labels_y) in enumerate(val_dataloader_loop):
            # forward pass: Feed inputs to the model and compute loss.
            # No need to compute gradients in validation phase (backpropagation)
            inputs_x, labels_y = inputs_x.to(cfg.DEVICE), labels_y.to(
                cfg.DEVICE
            )  # Move data to device (GPU if available)
            output = model(inputs_x)  # Feed inputs to the model to get predictions
            loss = loss_fn(output, labels_y)  # calculate loss
            losses.append(loss.item())

    # Compute mean loss over all batches of the validation data
    mean_loss = sum(losses) / len(losses)
    print(f"Val mean loss: {mean_loss}")
