import torch
import torchmetrics


def eval_model(model, val_loader, loss_fn, epochs,  device='cpu'):
    val_accuracies = []
    val_losses = []

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=model.get_classes()).to(device)

    model.to(device)
    model.eval()
    for epoch in range(epochs):
        epoch_val_loss = 0
        epoch_val_accuracy = 0
        with torch.no_grad():
            for batch_xs, batch_ys in val_loader:
                batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

                # Forward pass: Model Prediction
                outputs = model(batch_xs)

                # Loss
                loss = loss_fn(outputs, batch_ys)
                epoch_val_loss += loss.item() / 100

                # e.g. -> [0.3, 0.6, 0.1] -> [1]
                prediction = torch.argmax(outputs, dim=1)
                target = torch.argmax(batch_ys, dim=1)

                # Accuracy
                batch_accuracy = accuracy_metric(prediction, target)
                epoch_val_accuracy += batch_accuracy.item()

        epoch_val_loss /= len(val_loader)
        epoch_val_accuracy /= len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        # Epoch summary
        print(
            f"Epoch [{epoch + 1}/{epochs}] Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}"
        )

    print("Validation Finished!")

    return val_losses, val_accuracies


def predict(model, input_data, device='cpu'):
    model.to(device)
    model.eval()

    # torch.Size([100, 126]) -> torch.Size([1, 100, 126])
    input_data = input_data.unsqueeze(0).to(device)

    with torch.no_grad():
        # Prediction
        outputs = model(input_data).to(device)

    prediction = torch.argmax(outputs, dim=1).item()
    return prediction
