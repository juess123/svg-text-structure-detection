import torch
import pickle
from models.text_energy_model import TextEnergyModel


def load_model():

    model = TextEnergyModel(feature_dim=9)
    model.load_state_dict(torch.load("models/text_energy_binary.pth", weights_only=True))
    model.eval()

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler