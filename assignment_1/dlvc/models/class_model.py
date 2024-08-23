import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        ## TODO implement
        # Ensure the directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create the filename
        filename = "model.pth"
        if suffix:
            filename = f"model_{suffix}.pth"

        save_path = save_dir / filename

        # Save the model's state dictionary
        torch.save(self.net.state_dict(), save_path)

        print(f"Model saved to {save_path}")

    def load(self, path, model_type):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        path = Path(path) # convert string to Path object

        ## TODO implement
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # if cuda not available, load to cpu - otherwise problems might arise when the training was done a GPU, but testing is done on a CPU
        if torch.cuda.is_available():
            state_dict = torch.load(str(path))
        else:
            state_dict = torch.load(str(path), map_location=torch.device('cpu'))

        state_dict = self.remove_net_prefix(state_dict) 

        self.net.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

    @staticmethod
    def remove_net_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("net."):
                new_key = key.replace("net.", "", 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

class DeepClassifierWithDropout(DeepClassifier):
    def __init__(self, net: nn.Module, dropout_prob = 0.5):
        super().__init__(net)
        self.dropout = nn.Dropout(p = dropout_prob)

    def forward(self, x):
        x = super().forward(x)
        x = self.dropout(x)
        return x
