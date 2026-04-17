import torch
import numpy as np
import json
import pickle

from models.text_classifier_model import TextClassifierModel
from utils.feature_loader import load_features
from utils.scaler import StandardScaler


LABEL_MAP = {
    "text": 0,
    "art_text": 1,
    "non_text": 2,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def print_class_distribution(labels):
    print("\n===== Class Distribution =====")
    for class_idx in sorted(IDX_TO_LABEL.keys()):
        count = int((labels == class_idx).sum())
        print(f"{IDX_TO_LABEL[class_idx]:>10s}: {count}")


def compute_confusion_matrix(y_true, y_pred, num_classes=3):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t.long(), p.long()] += 1
    return cm


def print_metrics(y_true, y_pred):
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

    print("\n===== Confusion Matrix =====")
    header = "true\\pred".ljust(12)
    for i in range(3):
        header += f"{IDX_TO_LABEL[i]:>12s}"
    print(header)

    for i in range(3):
        row = f"{IDX_TO_LABEL[i]:<12}"
        for j in range(3):
            row += f"{cm[i, j].item():>12d}"
        print(row)

    print("\n===== Per-Class Metrics =====")
    for i in range(3):
        tp = cm[i, i].item()
        fn = cm[i, :].sum().item() - tp
        fp = cm[:, i].sum().item() - tp

        recall = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)

        print(
            f"{IDX_TO_LABEL[i]:>10s} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f}"
        )


def print_prediction_distribution(y_pred):
    print("\n===== Prediction Distribution =====")
    for class_idx in sorted(IDX_TO_LABEL.keys()):
        count = int((y_pred == class_idx).sum())
        print(f"{IDX_TO_LABEL[class_idx]:>10s}: {count}")


def print_hard_examples(data, y_true, y_pred, probs, topk=8):
    print("\n===== Hard Examples =====")

    confidence, _ = torch.max(probs, dim=1)

    # 1. 错误样本里，置信度最高的（模型很自信但错了）
    wrong_mask = y_true != y_pred
    wrong_indices = torch.where(wrong_mask)[0]

    if len(wrong_indices) > 0:
        wrong_conf = confidence[wrong_indices]
        sorted_wrong = wrong_indices[torch.argsort(wrong_conf, descending=True)]

        print(f"\n[Top {min(topk, len(sorted_wrong))} most confident WRONG samples]")
        for idx in sorted_wrong[:topk]:
            item = data[idx.item()]
            print(
                f"id={item.get('id', 'NO_ID')}, "
                f"true={IDX_TO_LABEL[y_true[idx].item()]}, "
                f"pred={IDX_TO_LABEL[y_pred[idx].item()]}, "
                f"conf={confidence[idx].item():.4f}"
            )
    else:
        print("\nNo wrong samples found.")

    # 2. 正确样本里，置信度最低的（虽然对了，但在边界上）
    correct_mask = y_true == y_pred
    correct_indices = torch.where(correct_mask)[0]

    if len(correct_indices) > 0:
        correct_conf = confidence[correct_indices]
        sorted_correct = correct_indices[torch.argsort(correct_conf, descending=False)]

        print(f"\n[Top {min(topk, len(sorted_correct))} lowest-confidence CORRECT samples]")
        for idx in sorted_correct[:topk]:
            item = data[idx.item()]
            print(
                f"id={item.get('id', 'NO_ID')}, "
                f"label={IDX_TO_LABEL[y_true[idx].item()]}, "
                f"conf={confidence[idx].item():.4f}"
            )


def train():
    # 1. 加载特征
    X = load_features("./data/dataset_features.json")

    # 2. 加载原始数据和标签
    with open("./data/dataset_features.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = np.array(
        [LABEL_MAP[item["label"]] for item in data],
        dtype=np.int64
    )

    print_class_distribution(labels)

    # 3. 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # 4. 建模
    model = TextClassifierModel(
        feature_dim=X.shape[1],
        num_classes=3
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    class_weights = torch.tensor([1.0, 1.3, 1.0], dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 5. 训练
    for epoch in range(600):
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y_tensor).float().mean().item()
            print(f"Epoch {epoch:4d} | Loss={loss.item():.4f} | Acc={acc:.4f}")

    # 6. 训练结束后做详细分析
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y_tensor).float().mean().item()

        print(f"\n===== Final Train Accuracy =====")
        print(f"Acc={acc:.4f}")

        print_prediction_distribution(pred)
        print_metrics(y_tensor, pred)
        print_hard_examples(data, y_tensor, pred, probs, topk=8)

    # 7. 保存
    torch.save(model.state_dict(), "./models/text_classifier_3class.pth")
    with open("./models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n3-class model and scaler saved.")


if __name__ == "__main__":
    train()