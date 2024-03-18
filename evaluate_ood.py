import numpy as np
import pyxis as px
import torch
import torch.utils.data
import pyxis.torch as pxt
from pytorch_ood.detector import OpenMax
from pytorch_ood.utils import OODMetrics, ToUnknown
from utils import read_paths 

paths_data = read_paths("./release/paths.json")
BASE_SAVE_PATH = paths_data["base_save_path"]  # Should point to the base folder of the codebase

INPUT_DIMS = 1000
METHOD_MNEMONIC = "naivedefinition"  # "seekdefinition" "naivedefinition"

DATASET = "spiders"  # "birds" "garments" "spiders" "ungulate" "fruit" "aircraft"

if DATASET == "garments":
    ds_names = ["garments", "furniture"]
    NUM_CLASSES = 59
elif DATASET == "birds":
    ds_names = ["birds", "huntingdog"]
    NUM_CLASSES = 26
elif DATASET == "spiders":
    ds_names = ["spiders", "butterflies"]
    NUM_CLASSES = 6
elif DATASET == "ungulate":
    ds_names = ["ungulate", "bear"]
    NUM_CLASSES = 17
elif DATASET == "fruit":
    ds_names = ["fruit", "musical_instrument"]
    NUM_CLASSES = 10
elif DATASET == "aircraft":
    ds_names = ["aircraft", "vessel"]
    NUM_CLASSES = 4
else:
    raise ValueError("Invalid dataset")

train_dataset = pxt.TorchDataset(f'{BASE_SAVE_PATH}datasets/imagenet_{DATASET}/imagenet_{ds_names[0]}_allclasses_allim_resnetnofilter')
test_dataset = pxt.TorchDataset(f'{BASE_SAVE_PATH}datasets/imagenet_{DATASET}/imagenet_{ds_names[0]}_allclasses_allvalim_resnetnofilter')
ood_dataset = pxt.TorchDataset(f'{BASE_SAVE_PATH}datasets/imagenet_{DATASET}/imagenet_{ds_names[1]}_allclasses_allvalim_resnetnofilter')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers = 0)
ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=64, shuffle=True, num_workers = 0)
all_loader = torch.utils.data.DataLoader(test_dataset + ood_dataset, batch_size=64, shuffle=True, num_workers = 0)

import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIMS, INPUT_DIMS)  # 256
        self.fc2 = nn.Linear(INPUT_DIMS, NUM_CLASSES)  # 512, 475
        self.preds = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.get_av(x)

    def get_av(self, x):
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def get_softmax(self, x):
        x = self.get_av(x)
        x = self.preds(x)
        return x

net = Classifier()

net.load_state_dict(torch.load(f"{BASE_SAVE_PATH}model_weights/{DATASET}_resnet"))

# ROC functions
import matplotlib.pyplot as plt

def thr_performance(thresh):
    global scores_ind, scores_ood
    # 'positive' is OOD=1, or, when the max logit is below a threshold
    fp = np.sum(scores_ind < thresh)
    tp = np.sum(scores_ood < thresh)

    tpr = tp / scores_ood.size
    fpr = fp / scores_ind.size

    return fpr, tpr


def plot_roc(segments, display=False):
    global scores_ind, scores_ood
    auroc = 0
    all_scores = np.concatenate((scores_ind, scores_ood))
    min = np.min(all_scores)
    max = np.max(all_scores)
    thr_list = np.linspace(min,max,segments, endpoint=True)

    fpr = []
    tpr = []
    old_x = None
    old_y = None
    for thr in thr_list:
        x, y = thr_performance(thr)
        fpr.append(x)
        tpr.append(y)
        if old_x is not None:
            auroc += ((y+old_y)*(x-old_x))/2
        old_x = x
        old_y = y

    if display:
        plt.figure(figsize=(4,4))
        plt.plot(fpr, tpr)
        plt.show()
    return auroc

    # Maximum Logit Score

# Computing scores
scores_ind = np.empty([0])
for batch in test_loader:
    x = batch['X'].float()
    y = batch['y'].long().squeeze()

    with torch.no_grad():
        values = torch.amax(net.get_av(x), dim=1).numpy().squeeze()
    scores_ind = np.concatenate((scores_ind, values))

scores_ood = np.empty([0])
for batch in ood_loader:
    x = batch['X'].float()
    y = batch['y'].long().squeeze()

    with torch.no_grad():
        values = torch.amax(net.get_av(x), dim=1).numpy().squeeze()
    scores_ood = np.concatenate((scores_ood, values))


a = plot_roc(1000)
print(f"AUROC MLS: {a}")

# Hierarchy performance
import pickle
import numpy as np
with open(f"{BASE_SAVE_PATH}ood_scores/scores_{ds_names[0]}_allclasses_allvalim_clip{METHOD_MNEMONIC}", "rb") as f:
    scores_ind = pickle.load(f)
    scores_ind = np.array([x[0]/x[1] for x in scores_ind])
with open(f"{BASE_SAVE_PATH}ood_scores/scores_{ds_names[1]}_allclasses_allvalim_clip{METHOD_MNEMONIC}", "rb") as f:
    scores_ood = pickle.load(f)
    scores_ood = np.array([x[0]/x[1] for x in scores_ood])


a = plot_roc(1000)
print(f"AUROC ConV {METHOD_MNEMONIC}: {a}")