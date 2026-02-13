import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==================== KNKNN ====================
def knknn_classifier(trainy, testy, trainx, testx, k=3, m=2, sigma=5, delta=0.1):
    num_train = trainx.shape[0]
    num_test = testx.shape[0]
    num_classes = 2

    predicted = np.zeros(num_test)
    memberships = np.zeros((num_test, num_classes))

    # Compute class centroids
    centroids = np.zeros((num_classes, trainx.shape[1]))
    for i in range(num_classes):
        class_indices = np.where(trainy == i)[0]
        if len(class_indices) > 0:
            centroids[i, :] = np.mean(trainx[class_indices, :], axis=0)

    # Neutrosophic parameters for training samples
    train_neutrosophic = np.zeros((num_train, 3))
    for i in range(num_train):
        dist_to_centroids = np.zeros(num_classes)
        for j in range(num_classes):
            dist = np.linalg.norm(trainx[i, :] - centroids[j, :])
            dist_to_centroids[j] = 2 * (1 - np.exp(-dist**2 / sigma**2))

        sorted_idx = np.argsort(dist_to_centroids)
        nearest1 = centroids[sorted_idx[0], :]
        nearest2 = centroids[sorted_idx[1], :]
        neutrosophic_point = (nearest1 + nearest2) / 2
        dist_to_neutrosophic = np.linalg.norm(trainx[i, :] - neutrosophic_point)

        ker_dist_to_centroids = 2 * (1 - np.exp(-dist_to_centroids**2 / sigma**2))
        ker_dist_to_neutrosophic = 2 * (1 - np.exp(-dist_to_neutrosophic**2 / sigma**2))

        denominator = (np.sum(ker_dist_to_centroids ** (-2/(m-1))) +
                       ker_dist_to_neutrosophic ** (-2/(m-1)) +
                       delta ** (-2/(m-1)))

        T = np.zeros(num_classes)
        for j in range(num_classes):
            T[j] = (ker_dist_to_centroids[j] ** (-2/(m-1))) / denominator

        I = (ker_dist_to_neutrosophic ** (-2/(m-1))) / denominator
        F = (delta ** (-2/(m-1))) / denominator

        train_neutrosophic[i, 0] = T[np.argmin(dist_to_centroids)]
        train_neutrosophic[i, 1] = I
        train_neutrosophic[i, 2] = F

    # Classify test samples
    for i in range(num_test):
        distances = np.zeros(num_train)
        for j in range(num_train):
            dist = np.linalg.norm(testx[i, :] - trainx[j, :])
            distances[j] = 2 * (1 - np.exp(-dist**2 / sigma**2))

        neighbors = np.argsort(distances)[:k]
        weights = distances[neighbors] ** (-1/(m-1))
        weights = np.where(np.isinf(weights), 1, weights)

        neighbor_labels = trainy[neighbors]
        neighbor_neutrosophic = train_neutrosophic[neighbors]

        class_memberships = np.zeros(num_classes)
        for j, nb in enumerate(neighbors):
            cls = int(neighbor_labels[j])
            T = neighbor_neutrosophic[j, 0]
            I = neighbor_neutrosophic[j, 1]
            F = neighbor_neutrosophic[j, 2]
            class_memberships[cls] += weights[j] * (T + I - F)

        total_weight = np.sum(weights)
        if total_weight > 0:
            class_memberships /= total_weight

        memberships[i, :] = class_memberships
        predicted[i] = np.argmax(class_memberships)

    predicted = np.clip(predicted, 0, 1)

    accuracy = accuracy_score(testy, predicted)
    precision = precision_score(testy, predicted, average='macro', zero_division=0)
    recall = recall_score(testy, predicted, average='macro', zero_division=0)
    f1 = f1_score(testy, predicted, average='macro', zero_division=0)
    cm = confusion_matrix(testy, predicted)

    return accuracy, precision, recall, f1, cm

# ==================== Data Loading and Experiment ====================
def load_feature_data(file_path, delimiter):
    data = pd.read_csv(file_path, header=None, sep=delimiter).values
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(float)
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    print(f"  Label distribution: Normal {(y == 0).sum()}, Fault {(y == 1).sum()}")
    return X, y

def run_experiment(X, y, dataset_name):
    print(f"{dataset_name} Dataset — KNKNN Classification Experiment")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} (Normal: {(y_train==0).sum()}, Fault: {(y_train==1).sum()})")
    print(f"Test set: {len(X_test)} (Normal: {(y_test==0).sum()}, Fault: {(y_test==1).sum()})")

    acc, prec, rec, f1, cm = knknn_classifier(
        y_train, y_test, X_train, X_test, k=3, m=2, sigma=5, delta=0.1
    )

    print(f"\nAccuracy: {acc:.4f} Precision: {prec:.4f} Recall: {rec:.4f} F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(f"[[TN={cm[0,0]} FP={cm[0,1]}] FN={cm[1,0]} TP={cm[1,1]}]")
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1, 'Confusion Matrix': cm}

# ==================== Main Program ====================
def main():

    # Fan 15
    print("\nFan 15")
    X15, y15 = load_feature_data('15wind.txt', delimiter='\t')
    run_experiment(X15, y15, "Fan 15")

    # Fan 21
    print("\nFan 21")
    X21, y21 = load_feature_data('21wind.txt', delimiter='\t')
    run_experiment(X21, y21, "Fan 21")

    # CWRU
    print("\nCWRU")
    Xcwru, ycwru = load_feature_data('CWRU_dataset.txt', delimiter='\t')
    run_experiment(Xcwru, ycwru, "CWRU")

if __name__ == "__main__":
    main()