import torch
import numpy as np
import json
from models.text_energy_model import TextEnergyModel
from utils.feature_loader import load_features, FEATURE_ORDER
from utils.scaler import StandardScaler

# 1️⃣ 加载数据
with open("./data/dataset_features.json") as f:
    data = json.load(f)

X = load_features("./data/dataset_features.json")

labels = np.array([
    1 if item["label"] == "text" else 0
    for item in data
], dtype=np.int32)

# 2️⃣ 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X)

# 3️⃣ 加载模型
model = TextEnergyModel(feature_dim=X.shape[1])
model.load_state_dict(torch.load("./models/text_energy_binary.pth"))
model.eval()

# 4️⃣ 计算 energy
with torch.no_grad():
    _, energy = model(X_tensor)

energy = energy.numpy()

# 5️⃣ 设定阈值
low_th  = 10
high_th = 15

correct = 0
wrong = 0
uncertain = 0

for e, y in zip(energy, labels):

    if e < low_th:
        pred = 1
    elif e > high_th:
        pred = 0
    else:
        uncertain += 1
        continue

    if pred == y:
        correct += 1
    else:
        wrong += 1

print("Total samples:", len(labels))
print("Correct:", correct)
print("Wrong:", wrong)
print("Uncertain:", uncertain)
print("Accuracy (excluding uncertain):",
      correct / (correct + wrong))