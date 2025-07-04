import torch
import torch.nn as nn
import joblib
import numpy as np

class WaistPredNet(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def predict(measurements):
    '''Predict waist circumference from body measurements'''

    # Load model
    model = WaistPredNet(input_features=5)
    model.load_state_dict(torch.load("waist_model.pt", map_location=torch.device('cpu')))
    model.eval()

    # Load scalers
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    # Convert input
    if isinstance(measurements, torch.Tensor):
        measurements = measurements.numpy()

    # Scale and predict
    X_scaled = scaler_X.transform(measurements)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_scaled = model(X_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_scaled)

    return torch.tensor(y_pred, dtype=torch.float32)
