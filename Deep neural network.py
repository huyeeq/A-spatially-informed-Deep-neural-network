# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
# Please fill in all the parameters and paths in this section.
CONFIG = {
    # -- Data and Paths --
    "data_path": r"E:\PATH\TO\YOUR\DIN.xlsx",
    "model_save_path": r"E:\PATH\TO\SAVE\dnn_model.pth",
    "target_column": "DIN",
    
    # -- Data Splitting and Preprocessing --
    "train_size": 0.75,
    
    # -- Label Distribution Smoothing (LDS) Settings --
    "use_lds": True,
    "lds_reweight_strategy": 'inverse',  # 'inverse' or 'sqrt_inv'
    "lds_kernel": 'gaussian',             # 'gaussian', 'triang', 'laplace'
    "lds_ks": 5,                          # Kernel size
    "lds_sigma": 2,                       # Sigma for gaussian/laplace kernel

    # -- Model Architecture --
    "input_size": 16,
    "hidden_layers": [256, 256, 32],      # List of hidden layer sizes
    "output_size": 1,
    "dropout_rate": 0.3,

    # -- Training Hyperparameters --
    "learning_rate": 0.001,
    "num_epochs": 1000,
    "batch_size": 32,
    "early_stopping_patience": 250,
}

# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================
def get_lds_kernel_window(kernel: str, ks: int, sigma: float) -> np.ndarray:
    """Generates a kernel window for Label Distribution Smoothing."""
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = np.zeros(ks)
        base_kernel[half_ks] = 1.
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
        return kernel_window / np.max(kernel_window)
    elif kernel == 'triang':
        return triang(ks)
    else:  # laplace
        laplace_func = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = np.array([laplace_func(x) for x in np.arange(-half_ks, half_ks + 1)])
        return kernel_window / np.max(kernel_window)

def prepare_weights(labels: np.ndarray, config: Dict) -> np.ndarray:
    """Prepares sample weights using optional Label Distribution Smoothing (LDS)."""
    reweight_strategy = config["lds_reweight_strategy"]
    
    value_counts = pd.Series(labels).value_counts().to_dict()
    
    if reweight_strategy == 'sqrt_inv':
        value_counts = {k: np.sqrt(v) for k, v in value_counts.items()}
    elif reweight_strategy == 'inverse':
        value_counts = {k: np.clip(v, 5, 1000) for k, v in value_counts.items()}
    
    num_per_label = np.array([value_counts.get(label, 1) for label in sorted(value_counts.keys())])

    if config["use_lds"]:
        lds_kernel_window = get_lds_kernel_window(config["lds_kernel"], config["lds_ks"], config["lds_sigma"])
        smoothed_counts = convolve1d(num_per_label, weights=lds_kernel_window, mode='constant')
        
        # Create a mapping from label value to its smoothed count
        label_to_smoothed_count = {label: smoothed_counts[i] for i, label in enumerate(sorted(value_counts.keys()))}
        
        # Assign weights based on smoothed counts
        weights = [1.0 / label_to_smoothed_count.get(label, 1) for label in labels]
    else:
        weights = [1.0 / value_counts.get(label, 1) for label in labels]
        
    # Normalize weights
    scaling = len(weights) / np.sum(weights)
    weights = np.array(weights, dtype=np.float32) * scaling
    return weights

def weighted_mse_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Calculates weighted Mean Squared Error loss."""
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    return torch.mean(loss)

def plot_losses(train_losses: List[float], val_losses: List[float]):
    """Plots training and validation losses over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================================================================
# 4. DATA HANDLING CLASS
# ==============================================================================
class DataHandler:
    """Handles loading, preprocessing, and splitting of the dataset."""
    def __init__(self, config: Dict):
        self.config = config

    def load_and_split_data(self) -> Tuple[pd.DataFrame, ...]:
        """Loads data and splits it into training and testing sets."""
        df = pd.read_excel(self.config["data_path"], index_col=0, header=0)
        
        if self.config["use_lds"]:
            labels = df[self.config["target_column"]].values
            weights = prepare_weights(labels, self.config)
            df['weights'] = weights
        else:
            df['weights'] = 1.0

        df_sorted = df.sort_values(by=self.config["target_column"])
        
        n = len(df_sorted)
        train_count = int(n * self.config["train_size"])
        
        train_indices = np.linspace(0, n - 1, train_count, dtype=int)
        test_indices = np.setdiff1d(np.arange(n), train_indices)

        train_set = df_sorted.iloc[train_indices]
        test_set = df_sorted.iloc[test_indices]
        
        feature_columns = df.drop(columns=[self.config["target_column"], 'weights']).columns.tolist()

        X_train = train_set[feature_columns].values
        X_test = test_set[feature_columns].values
        y_train = train_set[self.config["target_column"]].values
        y_test = test_set[self.config["target_column"]].values
        weights_train = train_set['weights'].values
        weights_test = test_set['weights'].values
        
        return X_train, X_test, y_train, y_test, weights_train, weights_test

    def get_dataloaders(self) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates PyTorch DataLoaders and test tensors."""
        X_train, X_test, y_train, y_test, w_train, w_test = self.load_and_split_data()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        w_train_tensor = torch.tensor(w_train, dtype=torch.float32).view(-1, 1)
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        w_test_tensor = torch.tensor(w_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, w_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        
        return train_loader, X_test_tensor, y_test_tensor, w_test_tensor

# ==============================================================================
# 5. DNN MODEL DEFINITION
# ==============================================================================
class DNN(nn.Module):
    """A flexible Deep Neural Network for regression."""
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, dropout_rate: float):
        super(DNN, self).__init__()
        
        layers = []
        layer_sizes = [input_size] + hidden_layers
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.output_layer(x)
        return x

# ==============================================================================
# 6. MODEL TRAINER AND EVALUATOR
# ==============================================================================
class ModelTrainer:
    """Handles the training, validation, and evaluation of the DNN model."""
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_loader, X_test, y_test, w_test):
        """Main training loop with early stopping."""
        X_test, y_test, w_test = X_test.to(self.device), y_test.to(self.device), w_test.to(self.device)
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            epoch_train_loss = 0.0
            for inputs, labels, weights in train_loader:
                inputs, labels, weights = inputs.to(self.device), labels.to(self.device), weights.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = weighted_mse_loss(outputs, labels, weights)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test)
                val_loss = weighted_mse_loss(val_outputs, y_test, w_test)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())

            print(f'Epoch [{epoch + 1}/{self.config["num_epochs"]}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config["model_save_path"])
                print(f"Validation loss improved. Model saved to {self.config['model_save_path']}")
            else:
                patience_counter += 1

            if patience_counter >= self.config["early_stopping_patience"]:
                print("Early stopping triggered.")
                break
        
        print("Training finished.")
        # Load the best model for final evaluation
        self.model.load_state_dict(torch.load(self.config["model_save_path"]))
        return train_losses, val_losses

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor, w_test: torch.Tensor) -> Dict[str, float]:
        """Evaluates the model and returns performance metrics."""
        X_test, y_test, w_test = X_test.to(self.device), y_test.to(self.device), w_test.to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test)
        
        y_test_np = y_test.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        w_test_np = w_test.cpu().numpy().flatten() if w_test is not None else None
        
        mae = mean_absolute_error(y_test_np, y_pred_np, sample_weight=w_test_np)
        mse = mean_squared_error(y_test_np, y_pred_np, sample_weight=w_test_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_np, y_pred_np, sample_weight=w_test_np)
        
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        
        print("\n--- Model Performance ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("-------------------------\n")
        
        return metrics

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # 1. Initialize data handler and load data
    data_handler = DataHandler(CONFIG)
    train_loader, X_test_tensor, y_test_tensor, weights_test_tensor = data_handler.get_dataloaders()
    
    # 2. Initialize the DNN model
    model = DNN(
        input_size=CONFIG["input_size"],
        hidden_layers=CONFIG["hidden_layers"],
        output_size=CONFIG["output_size"],
        dropout_rate=CONFIG["dropout_rate"]
    )
    
    # 3. Initialize the trainer and start training
    trainer = ModelTrainer(model, CONFIG)
    train_losses, val_losses = trainer.train(train_loader, X_test_tensor, y_test_tensor, weights_test_tensor)
    
    # 4. Evaluate the best model
    final_metrics = trainer.evaluate(X_test_tensor, y_test_tensor, weights_test_tensor)
    
    # 5. Visualize the training process
    plot_losses(train_losses, val_losses)
