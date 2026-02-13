import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.io import loadmat
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ==================== Neutrosophic Transformer ====================
class NeutrosophicTransformer(nn.Module):
    def __init__(self, input_dim=20, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.neutrosophic_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )

        self.base_threshold = nn.Parameter(torch.tensor(0.5))
        self.lambda_param = nn.Parameter(torch.tensor(0.1))

        self.residual_correction = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, d_model)
        )

        self.classifier = nn.Linear(d_model, 2)

    def compute_neutrosophic(self, x):
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        T = x_norm.mean(dim=1, keepdim=True)
        F = 1 - T
        I = torch.sqrt(T ** 2 + F ** 2)
        return T, F, I

    def dynamic_threshold(self, T, F, I):
        confidence = (T - F - I + 1) / 2
        threshold = self.base_threshold + self.lambda_param * (0.5 - confidence)
        return torch.sigmoid(threshold)

    def forward(self, x):
        batch_size = x.shape[0]
        embedded = self.embedding(x).unsqueeze(1) + self.pos_encoding
        transformer_out = self.transformer(embedded)
        transformer_out = transformer_out.squeeze(1)

        T, F, I = self.compute_neutrosophic(transformer_out)
        neutrosophic_features = torch.cat([T, F, I], dim=1)

        threshold = self.dynamic_threshold(T, F, I)
        residual = self.residual_correction(neutrosophic_features)
        final_output = transformer_out + residual
        classification_out = self.classifier(final_output)

        return final_output, classification_out, threshold, (T, F, I)

# ==================== Feature Extraction and Evaluation Tools ====================
def train_feature_extractor(model, X_train, y_train, epochs=50, lr=1e-5, device='cpu', verbose=True):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            _, output, _, _ = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    model.eval()
    return model

def extract_features(model, X, device='cpu'):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        features, _, _, _ = model(X_tensor)
    return features.cpu().numpy()

def evaluate_features(X_train_feat, X_test_feat, y_train, y_test, method_name="Neutrosophic Transformer"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{method_name} Feature Evaluation Results:")
    print(f"  Accuracy: {acc:.4f} Precision: {pre:.4f} Recall: {rec:.4f} F1-score: {f1:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    return {'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1-score': f1, 'Confusion Matrix': cm}

def save_features(X_features, y_labels, filename):
    idx0 = np.where(y_labels == 0)[0]
    idx1 = np.where(y_labels == 1)[0]
    X_sorted = np.vstack([X_features[idx0], X_features[idx1]])
    y_sorted = np.concatenate([y_labels[idx0], y_labels[idx1]])
    dataset = np.column_stack((y_sorted, X_sorted))
    np.savetxt(filename, dataset, delimiter='\t', fmt='%.6f')
    print(f"  Feature dataset saved to: {filename}")
    print(f"  Dataset shape: {dataset.shape}, Normal samples: {len(idx0)}, Fault samples: {len(idx1)}")

# ==================== Fan Dataset Experiments (15, 21) ====================
def run_fan_experiment(file_path, fan_no, device='cpu'):
    print(f"Fan {fan_no} Dataset Feature Extraction Experiment")
    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
    except:
        data = pd.read_csv(file_path, sep=' ', header=None)
    print(f"Data shape: {data.shape}")
    X = data.iloc[:, 1:].values.astype(np.float32)
    y = data.iloc[:, 0].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training set labels: Normal: {(y_train==0).sum()}, Fault: {(y_train==1).sum()}")
    print(f"Test set labels: Normal: {(y_test==0).sum()}, Fault: {(y_test==1).sum()}")

    input_dim = X.shape[1]
    model = NeutrosophicTransformer(input_dim=input_dim).to(device)

    print("\nTraining Neutrosophic Transformer...")
    model = train_feature_extractor(model, X_train, y_train, epochs=50, device=device)

    X_train_feat = extract_features(model, X_train, device)
    X_test_feat = extract_features(model, X_test, device)

    evaluate_features(X_train_feat, X_test_feat, y_train, y_test)

# ==================== CWRU Bearing Dataset Experiment ====================
def run_cwru_experiment(device='cpu'):
    print("CWRU Bearing Dataset Feature Extraction Experiment")
    NAME2NUM = {'97.mat': 'X097', '118.mat': 'X118', '130.mat': 'X130', '105.mat': 'X105'}
    WINDOW, STEP = 8, 32

    def load_de_time(path):
        data = loadmat(path)
        basename = os.path.basename(path)
        number = NAME2NUM[basename]
        key = f'{number}_DE_time'
        return data[key].flatten()

    files = ['97.mat', '118.mat', '130.mat', '105.mat']
    labels = [0, 1, 1, 1]
    data_seq = [load_de_time(f) for f in files]

    def sliding_window(seq, win, step):
        return np.stack([seq[i:i+win] for i in range(0, len(seq)-win+1, step)])

    X_windows = []
    y_windows = []
    for seq, lab in zip(data_seq, labels):
        windows = sliding_window(seq, WINDOW, STEP)
        X_windows.append(windows)
        y_windows.append(np.full(len(windows), lab))

    X = np.concatenate(X_windows, axis=0).astype(np.float32)
    y = np.concatenate(y_windows, axis=0).astype(int)

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution: Normal samples: {(y==0).sum()}, Fault samples: {(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training set labels: Normal: {(y_train==0).sum()}, Fault: {(y_train==1).sum()}")

    input_dim = X.shape[1]
    model = NeutrosophicTransformer(input_dim=input_dim).to(device)

    print("\nTraining Neutrosophic Transformer...")
    model = train_feature_extractor(model, X_train, y_train, epochs=50, device=device)

    X_train_feat = extract_features(model, X_train, device)
    X_test_feat = extract_features(model, X_test, device)

    evaluate_features(X_train_feat, X_test_feat, y_train, y_test)

    X_all_feat = np.vstack([X_train_feat, X_test_feat])
    y_all = np.concatenate([y_train, y_test])
    save_features(X_all_feat, y_all, 'CWRU_dataset.txt')

# ==================== Fan Single Feature Importance Selection and Dataset Saving ====================
def run_fan_single_feature_selection(file_path, fan_no, top_k=8, device='cpu'):
    print(f"Fan {fan_no} Single Feature Importance Selection (select top {top_k} features)")
    # Load data (consistent with main experiment)
    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
    except:
        data = pd.read_csv(file_path, sep=' ', header=None)
    X = data.iloc[:, 1:].values.astype(np.float32)
    y = data.iloc[:, 0].values.astype(int)
    n_features = X.shape[1]

    # Fixed split (consistent with main experiment)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    importance_scores = np.zeros(n_features)
    print(f"\nEvaluating importance of {n_features} features (training a model for each feature individually)...")
    for i in range(n_features):
        # Take the i-th feature, reshape to (-1,1)
        X_train_i = X_train[:, i].reshape(-1, 1)
        X_test_i = X_test[:, i].reshape(-1, 1)

        # Initialize model, input_dim=1
        model = NeutrosophicTransformer(input_dim=1).to(device)

        # Train model (few epochs, no logging)
        model = train_feature_extractor(model, X_train_i, y_train, epochs=50, lr=1e-5,
                                        device=device, verbose=False)

        # Test: use model's own classification head
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test_i).to(device)
        with torch.no_grad():
            _, output, _, _ = model(X_test_tensor)
            y_pred = torch.argmax(output, dim=1).cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        importance_scores[i] = acc

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{n_features} features")

    # Select top_k features with highest importance
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]  # descending order
    print(f"\nFeature importance ranking (top {top_k}):")
    for rank, idx in enumerate(top_indices):
        print(f"  Rank {rank+1}: Feature {idx+1}, Accuracy: {importance_scores[idx]:.4f}")

    X_selected = X[:, top_indices]  # top_k features for all samples

    # Save to file
    filename = f'{fan_no}wind.txt'
    # Sort by class (class 0 first, then class 1)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    X_sorted = np.vstack([X_selected[idx0], X_selected[idx1]])
    y_sorted = np.concatenate([y[idx0], y[idx1]])
    dataset = np.column_stack((y_sorted, X_sorted))
    np.savetxt(filename, dataset, delimiter='\t', fmt='%.6f')

    print(f"\nSelected top {top_k} features dataset saved to: {filename}")
    print(f"Dataset shape: {dataset.shape}, Normal samples: {len(idx0)}, Fault samples: {len(idx1)}")
    print(f"Selected original column indices: {top_indices + 1}")  # convert to 1-based for readability

    return top_indices, importance_scores

# ==================== Main Program ====================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Fan 15
    print(f"{'=' * 60}")
    run_fan_experiment('15raw.txt', fan_no=15, device=device)

    # Fan 21
    print(f"{'=' * 60}")
    run_fan_experiment('21raw.txt', fan_no=21, device=device)

    # CWRU Bearing
    print(f"{'=' * 60}")
    run_cwru_experiment(device=device)

    # Fan 15 single feature selection
    print(f"{'=' * 60}")
    run_fan_single_feature_selection('15raw.txt', fan_no=15, top_k=8, device=device)

    # Fan 21 single feature selection
    print(f"{'=' * 60}")
    run_fan_single_feature_selection('21raw.txt', fan_no=21, top_k=8, device=device)

if __name__ == "__main__":
    main()