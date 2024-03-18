import numpy as np
import pyxis as px
import torch
import torch.utils.data
import pyxis.torch as pxt
from pytorch_ood.detector import OpenMax
from pytorch_ood.utils import OODMetrics, ToUnknown
from utils import read_paths

torch.manual_seed(0)

paths_data = read_paths("./release/paths.json")
BASE_SAVE_PATH = paths_data["base_save_path"]
INPUT_DIMS = 1000
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

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1*1e-4)
n_epochs = 300

for epoch in range(n_epochs):  # 100
    full_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['X'].float()
        labels = data['y'].long().squeeze()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.get_softmax(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step(full_loss)

        # print statistics
        curr_loss = loss.item()
        running_loss += curr_loss
        full_loss += curr_loss
        if i % 99 == 98:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {n_epochs}] loss: {running_loss / 99:.4f}')
            running_loss = 0.0
    # scheduler.step(full_loss)
    full_loss = 0

print('Finished Training')

top_n = min(5, NUM_CLASSES)

# Test set performance
net.eval()
ok_1 = 0
ok_5 = 0
total = 0
for i, data in enumerate(test_loader, 0):
    inputs = data['X'].float()
    labels = data['y'].long().squeeze().numpy()
    out = net.get_softmax(inputs).detach().numpy()
    sorted_out = np.argsort(out, axis=1)[:, ::-1]
    # Top 1
    top_1 = sorted_out[:,0]
    ok_1 += np.sum(top_1==labels)
    # Top 5
    ok_5 += np.sum(np.any(sorted_out[:, 0:top_n] == np.tile(labels.reshape(-1,1), (1,top_n)), axis=1))
    total += inputs.shape[0]

print(ok_1/total)
print(ok_5/total)

torch.save(net.state_dict(), f"{BASE_SAVE_PATH}model_weights/{DATASET}_resnet")