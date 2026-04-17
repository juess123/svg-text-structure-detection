import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from models.text_classifier_model import TextEnergyModel
from utils.feature_loader import load_features
from utils.scaler import StandardScaler


# =========================
# 1️⃣ 加载数据
# =========================

with open("./data/dataset_features.json") as f:
    data = json.load(f)

X = load_features("./data/dataset_features.json")

labels = np.array([
    1 if item["label"] == "text" else 0
    for item in data
], dtype=np.int32)


# =========================
# 2️⃣ 特征标准化
# =========================

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)


# =========================
# 3️⃣ 加载模型
# =========================

model = TextEnergyModel(feature_dim=X.shape[1])
model.load_state_dict(torch.load("./models/text_energy_binary.pth"))
model.eval()


# =========================
# 4️⃣ 计算 energy
# =========================

with torch.no_grad():
    _, energy = model(X_tensor)

energy = energy.numpy()


# =========================
# 5️⃣ 分离 text / non-text energy
# =========================

text_energy = energy[labels == 1]
non_energy = energy[labels == 0]

print("\n===== Energy Statistics =====")

print("Text mean :", text_energy.mean())
print("Text max  :", text_energy.max())

print("Non mean  :", non_energy.mean())
print("Non min   :", non_energy.min())


# =========================
# 6️⃣ 自动计算阈值
# =========================

text_p95 = np.percentile(text_energy, 95)
non_p05 = np.percentile(non_energy, 5)

mid = (text_p95 + non_p05) / 2

low_th = mid - 2
high_th = mid + 2

print("\n===== Threshold =====")
print("text_p95 :", text_p95)
print("non_p05  :", non_p05)
print("low_th   :", low_th)
print("high_th  :", high_th)


# =========================
# 7️⃣ 分类评估
# =========================

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


print("\n===== Evaluation =====")

print("Total samples:", len(labels))
print("Correct:", correct)
print("Wrong:", wrong)
print("Uncertain:", uncertain)

if correct + wrong > 0:
    print("Accuracy (excluding uncertain):",
          correct / (correct + wrong))


# =========================
# 8️⃣ 可视化 energy 分布
# =========================
plt.figure(figsize=(8,5))

plt.hist(text_energy, bins=50, alpha=0.6, label="Text")
plt.hist(non_energy, bins=50, alpha=0.6, label="Non-text")

plt.axvline(low_th, color="green", label="low_th")
plt.axvline(high_th, color="red", label="high_th")

plt.xlim(0,80)

plt.xlabel("Energy")
plt.ylabel("Count")
plt.title("Energy Distribution")

plt.legend()
plt.savefig("energy_distribution.png", dpi=300)