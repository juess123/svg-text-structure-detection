import torch
import numpy as np
import json
import pickle
from models.text_energy_model import TextEnergyModel
from utils.feature_loader import load_features
from utils.scaler import StandardScaler

def train():

    # 1️⃣ 加载特征
    X = load_features("./data/dataset_features.json")

    # 2️⃣ 加载 label
    with open("./data/dataset_features.json") as f:
        data = json.load(f)

    labels = np.array([
        1 if item["label"] == "text" else 0
        for item in data
    ], dtype=np.float32)

    # 3️⃣ 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(labels)

    # 4️⃣ 建模
    model = TextEnergyModel(feature_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    margin = 15.0   # 可以调

    # 5️⃣ 训练
    for epoch in range(1500):

        _, energy = model(X_tensor)

        text_mask = y_tensor == 1
        non_mask  = y_tensor == 0

        text_energy = energy[text_mask]
        non_energy  = energy[non_mask]

        # text 要小
        loss_text = text_energy.mean()

        # non_text 要大
        loss_non = torch.clamp(margin - non_energy, min=0).mean()

        loss = loss_text + loss_non

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "./models/text_energy_binary.pth")
    with open("./models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # 训练结束后查看分布
    with torch.no_grad():
        _, energy = model(X_tensor)

        text_energy = energy[y_tensor == 1]
        non_energy  = energy[y_tensor == 0]

        print("\n===== Final Energy Distribution =====")
        print("Text mean :", text_energy.mean().item())
        print("Text max  :", text_energy.max().item())
        print("Non mean  :", non_energy.mean().item())
        print("Non min   :", non_energy.min().item())
        w = torch.nn.functional.softplus(model.raw_w)
        print("w:", w.data)
    print("Binary model and scaler saved.")


if __name__ == "__main__":
    train()