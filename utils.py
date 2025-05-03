import json
import torch
from typing import Dict
from model import MatchPredictorFCNN  # Adjust import if needed

def load_config(path: str = "config.json") -> Dict:
    """Loads configuration values from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def loadModel(config: Dict) -> MatchPredictorFCNN:
    """Initializes and loads a trained model from weights."""
    model = MatchPredictorFCNN(input_size=config["input_size"])
    model.load_state_dict(torch.load(config["model_path"], map_location='cpu'))
    model.eval()
    return model
