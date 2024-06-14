import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import sys
sys.path.append("/root/Speech2Intent/s2i-baselines")

import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from trainer_hubert import LightningModel
from dataset import S2IMELDataset, collate_fn

# Load dataset
dataset = S2IMELDataset(
    csv_path="/home/tssdsilveira/corporas/speech-to-intent/test.csv",
    wav_dir_path="/home/tssdsilveira/corporas/speech-to-intent/",
)

# Load model checkpoint
model = LightningModel.load_from_checkpoint("checkpoints/hubert-sslepoch=12.ckpt")
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate model
trues = []
preds = []
print(dir(model))

for module in model.modules():
    if isinstance(module, nn.Conv1d):
        print(module.weight.shape)
for module in model.modules():
    if isinstance(module, nn.Conv1d):
        module.kernel_size = (2,)

import numpy as np
trues = []
preds = []
with torch.no_grad():
    for x, label in tqdm(dataset):
        x_tensor = x.view(3000, 80)
        x_tensor = x_tensor.transpose(0, 1)

        # Forward pass
        y_hat_l = model(x_tensor)

        # Calculate predictions
        probs = F.softmax(y_hat_l, dim=1).squeeze(0)
        pred = probs.argmax(dim=0).cpu().numpy().astype(int)

        trues.append(label)
        preds.append(pred)

trues = np.array(trues)
preds = np.array(preds)

# Convert preds to 1D array of class labels
preds_labels = preds.argmax(axis=1)

print("trues shape:", trues.shape)
print("preds_labels shape:", preds_labels.shape)

print(f"Accuracy Score = {accuracy_score(trues, preds_labels)}")
print(f"F1-Score = {f1_score(trues, preds_labels, average='weighted')}")