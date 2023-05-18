import torch
from d3s.d3s.datasets import ImageNet
import torchvision.transforms as T
import timm
import tqdm
import torch.nn as nn
import ssl
from collections import defaultdict
import pdb
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

from feature_extractor import feature_extractor

indexes = torch.load("imagenet_train_30_indexes.pt")
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = T.Compose(
    [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
train_imagenet = ImageNet(split="train", transform=train_transform)
train_labels = torch.load("train_labels.pt")
m = defaultdict(lambda: [])
wanted_indexes = []
wanted_labels = ["407", "670", "283", "200", "831", "765", "703", "414", "956", "950"]
class_names = ['ambulance', 'motor scooter', 'Persian cat', 'Tibetan terrier', 'studio couch', 'rocking chair',
               'park bench', 'backpack', 'custard apple', 'orange']
for i in range(len(train_labels)):
    if str(train_labels[i]) in wanted_labels:
        m[str(train_labels[i])].append(i)
        wanted_indexes.append(i)

model_category = "clip"
model_name = "RN50x4"
feats_train = feature_extractor(model_category, model_name, train_imagenet, wanted_indexes)
torch.save(feats_train, f'feats_train_{model_category}_{model_name}.pt')
# feats_train = torch.load("feats_train_resnet18.pt")

feature_reps = defaultdict()
# pdb.set_trace()
for i in m.keys():
    summ = torch.zeros(640).cuda()
    for j in m[i]:
        summ = torch.add(summ, feats_train[wanted_indexes.index(j)])
    summ = torch.div(summ, len(m[i]))
    feature_reps[i] = summ

the_diffs = [[0.0 for j in range(10)] for i in range(10)]
row = 0
# pdb.set_trace()
for i in feature_reps.keys():
    col = 0
    for j in feature_reps.keys():
        dist = (feature_reps[i] - feature_reps[j]).pow(2).sum().sqrt().item()
        the_diffs[row][col] = dist
        col += 1
    row += 1

fig, ax = plt.subplots(figsize=(15, 20))

g = sns.heatmap(np.array(the_diffs),
                linewidths=.5,
                ax=ax)
g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
g.set_xticklabels(g.get_yticklabels(), rotation=90, fontsize=8)
# passing a list is fine, no need to convert to tuples
ax.set_xticklabels(class_names, )
ax.set_yticklabels(class_names)
plt.savefig(f'./diffs_{model_category}_{model_name}.png')
