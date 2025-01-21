import torchmetrics
import torch


def train_model(model, train_loader, optimizer, loss_fn, epochs, device='cpu'):
    train_accuracies = []
    train_losses = []

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=model.get_classes()).to(device)

    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        for batch_idx, (batch_xs, batch_ys) in enumerate(train_loader):
            batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass: Model Prediction
            outputs = model(batch_xs)

            # e.g. -> [0.3, 0.6, 0.1] -> [1]
            prediction = torch.argmax(outputs, dim=1).float()
            target = torch.argmax(batch_ys, dim=1).float()

            # Loss
            loss = loss_fn(outputs, batch_ys)

            # Backward pass: Compute Gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            epoch_train_loss += loss.item() / 100

            # Accuracy
            batch_accuracy = accuracy_metric(prediction, target)
            epoch_train_accuracy += batch_accuracy.item()

            # Display progress every 100 iterations
            # if (batch_idx + 1) % 100 == 0:
            #     print(f"Epoch [{epoch + 1}/{self.n_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy.item():.4f}")

        epoch_train_loss /= len(train_loader)
        epoch_train_accuracy /= len(train_loader)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Epoch summary
        print(
            f"Epoch [{epoch + 1}/{epochs}] Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}")

    print("Training Finished!")

    return train_losses, train_accuracies
