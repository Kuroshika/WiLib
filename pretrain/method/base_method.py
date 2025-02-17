
from abc import abstractmethod
from tokenize import String
from datetime import datetime

import torch

# from model.sensefi import self_supervised
import os


class BaseMethod:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = None
        self.backbone = None
        self.projection_head = None
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.backbone.parameters(), "lr": 0.06},
                {"params": self.projection_head.parameters(), "lr": 0.06},
            ],
            weight_decay=1e-5,
        )
        self.scheduler = None
        self.total_loss = 0
        self.augmentation = None

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def predict():
        pass

    def save_model(self, is_best: bool = False, epoch: int = 0, model_save_path: str = None):
        if not model_save_path or model_save_path.strip() == "":
            raise ValueError("The model_save_path cannot be empty or None.")

        # Ensure the directory exists
        os.makedirs(model_save_path, exist_ok=True)

        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        current_backbone = self.backbone.__class__.__name__

        # Define the model name based on whether it's the best model or not
        model_suffix = "best" if is_best else "last"
        model_name = f"epoch[{epoch}]_{model_suffix}.pt"

        # Construct the full save path
        save_path = os.path.join(model_save_path, f"{current_time}_[{current_backbone}]_{model_name}")

        try:
            torch.save(self.backbone.state_dict(), save_path)
            print(f"Model successfully saved to {save_path}")
        except Exception as e:
            print(f"Failed to save model to {save_path}: {e}")
