import json
import torch
from typing import Dict
from model import MatchPredictorFCNN  # Adjust import if needed

def loadConfig(path: str = "config.json") -> Dict:
    """Loads configuration values from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)
    
def save_config(config: dict):
    try:
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Failed to save config: {e}")

def loadModel(config: Dict) -> MatchPredictorFCNN:
    """Initializes and loads a trained model from weights."""
    model = MatchPredictorFCNN(input_size=config["input_size"])
    model.load_state_dict(torch.load(config["model_path"], map_location='cpu'))
    model.eval()
    return model


