import torch
import numpy as np
import json
from models.text_classifier_model import TextEnergyModel
from utils.feature_loader import FEATURE_ORDER
from utils.scaler import StandardScaler
from utils.feature_loader import load_features

# 1️⃣ 重新加载训练数据，用于fit scaler
X_train = load_features("./data/dataset_features.json")

scaler = StandardScaler()
scaler.fit(X_train)

# 2️⃣ 加载测试数据
with open("./data/test_features.json") as f:
    test_data = json.load(f)

# 构造测试矩阵
X_test = []
for item in test_data:
    feat = item["features"]
    X_test.append([feat[k] for k in FEATURE_ORDER])

X_test = np.array(X_test, dtype=np.float32)
X_test = scaler.transform(X_test)

X_test_tensor = torch.tensor(X_test)

# 3️⃣ 加载模型
model = TextEnergyModel(feature_dim=X_test.shape[1])
model.load_state_dict(torch.load("./models/text_energy_binary.pth"))
model.eval()

# 4️⃣ 计算 energy
with torch.no_grad():
    _, energy = model(X_test_tensor)

for i, e in enumerate(energy):
    print(f"Path {i} energy: {e.item():.4f}")