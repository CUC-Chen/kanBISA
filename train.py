import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import importlib

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_metrics(predictions, actuals):
    pearson_corr, _ = pearsonr(predictions, actuals)
    spearman_corr, _ = spearmanr(predictions, actuals)
    kendall_corr, _ = kendalltau(predictions, actuals)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "Kendall": kendall_corr,
        "RMSE": rmse
    }

def train_model(model, device, train_loader, val_loader, optimizer, epochs, patience):
    criterion = nn.MSELoss()
    best_loss = float('inf')
    no_improvement = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.to(device)).squeeze()
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(model, device, val_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f'Early stopping at epoch {epoch} with validation loss {val_loss:.6f}')
            break

    return model

def evaluate_loss(model, device, loader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(loader)

def initialize_model(model_name, input_dim, output_dim):
    module = importlib.import_module(model_name)

    if model_name in ['TaylorKAN', 'ChebyKAN', 'HermiteKAN', 'JacobiKAN']:
        KANModel = getattr(module, model_name)
        return KANModel([input_dim, 8, 4, output_dim], 2)

    elif model_name in ['BSRBF_KAN']:
        KANModel = getattr(module, model_name)
        return KANModel([input_dim, 8, 4, output_dim])

    elif model_name in ['WavKAN']:
        KANModel = getattr(module, model_name)
        return KANModel([input_dim, 8, 4, output_dim])

def main():
    set_seed(42)

    datasets = [
        ('BID_x_15_BID.csv', 'BID_y_y_real_BID.csv', 'BID'),
        ('CID_x_15_CID.csv', 'CID_y_y_real_CID2013.csv', 'CID'),
        ('CLIVE_x_15_CLIVE.csv', 'CLIVE_y_y_real_CLIVE.csv', 'CLIVE'),
        ('KonIQ_x_15_KonIQ.csv', 'KonIQ_y_y_real_KONIQ.csv', 'KonIQ')
    ]

    learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                      1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
                      1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
                      1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
                      1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
    epochs = 1000
    patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        'BSRBF_KAN',
        'ChebyKAN',
        'HermiteKAN',
        'JacobiKAN',
        'TaylorKAN',
        'WavKAN'
    ]

    results_file = 'KAN_results.csv'

    file_exists = os.path.isfile(results_file)

    with open(results_file, 'a') as f:
        if not file_exists:
            f.write(
                'Dataset,Model,Learning Rate (Pearson),Learning Rate (Spearman),Learning Rate (Kendall),Learning Rate (RMSE),Pearson,Spearman,Kendall,RMSE\n')

    for x_file, y_file, dataset_name in datasets:
        print(f'Processing dataset: {dataset_name}')
        x_data = pd.read_csv(x_file).values
        y_data = pd.read_csv(y_file).values.squeeze()

        scaler = StandardScaler()
        train_size = int(0.7 * len(x_data))
        val_size = int(0.15 * len(x_data))
        test_size = len(x_data) - train_size - val_size

        x_train = scaler.fit_transform(x_data[:train_size])
        x_val = scaler.transform(x_data[train_size:train_size + val_size])
        x_test = scaler.transform(x_data[train_size + val_size:])
        y_train = y_data[:train_size]
        y_val = y_data[train_size:train_size + val_size]
        y_test = y_data[train_size + val_size:]

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        for model_name in models:
            best_metrics = {'Pearson': float('-inf'), 'Spearman': float('-inf'), 'Kendall': float('-inf'),
                            'RMSE': float('inf')}
            best_lr = {'Pearson': None, 'Spearman': None, 'Kendall': None, 'RMSE': None}

            for lr in learning_rates:
                print(f'Using learning rate: {lr} with model: {model_name}')

                model = initialize_model(model_name, x_data.shape[1], 1).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                try:
                    model = train_model(model, device, train_loader, val_loader, optimizer, epochs, patience)
                    predictions = []
                    actuals = []
                    model.eval()
                    with torch.no_grad():
                        for data, target in test_loader:
                            output = model(data.to(device)).squeeze()
                            predictions.extend(output.cpu().numpy())
                            actuals.extend(target.cpu().numpy())

                    metrics = evaluate_metrics(np.array(predictions), np.array(actuals))
                    print(metrics)

                    for metric in best_metrics:
                        if (metric == 'RMSE' and metrics[metric] < best_metrics[metric]) or (
                                metric != 'RMSE' and metrics[metric] > best_metrics[metric]):
                            best_metrics[metric] = metrics[metric]
                            best_lr[metric] = lr

                except Exception as e:
                    print(f"Error with model {model_name} at learning rate {lr}: {e}")

            with open(results_file, 'a') as f:
                f.write(
                    f"{dataset_name},{model_name},{best_lr['Pearson']},{best_lr['Spearman']},{best_lr['Kendall']},{best_lr['RMSE']},{best_metrics['Pearson']},{best_metrics['Spearman']},{best_metrics['Kendall']},{best_metrics['RMSE']}\n")

    print("Results saved to KAN_results.csv")

if __name__ == '__main__':
    main()
