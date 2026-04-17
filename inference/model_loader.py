import torch
import pickle
from models.text_classifier_model import TextClassifierModel
from utils.feature_loader import load_features


def load_model():
    X = load_features("./data/dataset_features.json")
    feature_dim = X.shape[1]

    model = TextClassifierModel(feature_dim=feature_dim, num_classes=3)
    model.load_state_dict(torch.load("models/text_classifier_3class.pth", weights_only=True))
    model.eval()

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler