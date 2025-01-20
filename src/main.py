import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from train.train import train_model
from train.eval import predict, eval_model
from utils.plotting import plot_training_progress, plot_learning_curve
from utils.loading_data import load_mocap_dataset_with_h5, get_hyper_parameters, get_hyper_parameters_data, load_data
from utils.util_methods import add_weight, get_classes, get_number_filename_to_save
from utils.processing_data import process_mocap_data, preparing_mocap_dataset

from model.LSTM import LSTM
from model.LSTM_RNN import LSTM_RNN
from model.GRU import GRU
from model.TransformerHAR import TransformerHAR

if __name__ == '__main__':
    # Load hyper parameters
    hyper_parameters = get_hyper_parameters_data()
    #   Dataset Parameters
    segment_size = get_hyper_parameters(hyper_parameters, "data", "segment_size")
    #   Model Structure
    epochs = get_hyper_parameters(hyper_parameters, "training", "epochs")
    n_hidden = get_hyper_parameters(hyper_parameters, "structure", "n_hidden")
    #   Training Data
    batch_size = get_hyper_parameters(hyper_parameters, "training", "batch_size")
    learning_rate = get_hyper_parameters(hyper_parameters, "training", "learning_rate")

    # Load dataset
    x, y = load_mocap_dataset_with_h5()
    x_data, y_data = process_mocap_data(x, y, segment_size)
    n_classes = get_classes(y)

    print(f"x data shape= {x_data.shape}")
    print(f"y data shape= {y_data.shape}")

    # Split data for train (76%) and validation (23%)
    train_size = int(len(x) * 0.76)
    x_train, x_val = x_data[:train_size], x_data[train_size:]
    y_train, y_val = y_data[:train_size], y_data[train_size:]

    # n_steps -> n timesteps per series
    # n_input -> n input parameters per timestep
    _, n_steps, n_input = x_train.shape
    print(f"n_steps={n_steps}")
    print(f"n_input={n_input}")

    # Model
    # model = LSTM()
    # model = LSTM_RNN()
    # model = GRU()
    model = TransformerHAR()
    model.init_hyper_parameters(n_steps, n_input, n_hidden, n_classes)
    model.init_model()
    # Display model summary
    print(model)

    # Select device: cpu or cuda if available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(device)

    # Weight for loss
    weight = add_weight(torch.tensor(y[:train_size]))
    # Cross-entropy Loss
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Loader
    train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)
    test_loader = data.DataLoader(data.TensorDataset(x_val, y_val), shuffle=True, batch_size=batch_size)

    # Train Model
    train_accuracy, train_loss = train_model(model, train_loader, optimizer, criterion, epochs, device=device)
    # Uncomment to see only train curve
    # plot_learning_curve(train_accuracy, train_loss)

    # Loss for validation
    criterion = nn.CrossEntropyLoss().to(device)
    # Validation
    test_accuracy, test_loss = eval_model(model, test_loader, criterion, epochs, device=device)
    # Uncomment to see only test curve
    # plot_learning_curve(test_accuracy, test_loss)

    plot_training_progress(train_loss, train_accuracy, test_loss, test_accuracy)

    # Save Model
    model.save_model()

    # Prediction part for test dataset: Save a csv file with all the predictions

    # Load test mocap dataset
    config_path = os.path.join(os.getcwd(), "..", "data/dataset")
    x_test = load_data(config_path, "test_mocap.h5")
    x_test = preparing_mocap_dataset(x_test, segment_size)
    x_test = np.array(x_test)
    test_data_tensor = torch.tensor(x_test, dtype=torch.float32)

    # Prediction
    predictions = []
    for i, x in enumerate(test_data_tensor):
        prediction = predict(model, x, device)
        print(i, prediction)
        predictions.append((i + 1, prediction))

    # Save prediction
    prediction_path = os.path.join(os.getcwd(), "..", "data/prediction")
    prediction_name = f"prediction_{get_number_filename_to_save(prediction_path, 'prediction')}"
    with open(f"{os.path.join(prediction_path, prediction_name)}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Label"])
        writer.writerows(predictions)

    print(f"CSV file '{prediction_name}.csv' created successfully.")
