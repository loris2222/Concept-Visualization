from nltk.corpus import wordnet as wn
import os
from language_utils import ConceptModel, WordModel
import clip
from utils import Embedding, read_paths
import torch
from PIL import Image
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F, InterpolationMode
from PIL import ImageFilter
import pyxis as px
import numpy as np
import wordnet_utils
import torch
import networkx as nx
from tqdm import tqdm

paths_data = read_paths("./release/paths.json")

IMAGENET_TRAIN_PATH = paths_data["imagenet_train_path"]
IMAGENET_VAL_PATH = paths_data["imagenet_val_path"]
DICT_FILE_PATH = paths_data["dict_file_path"]
BASE_SAVE_PATH = paths_data["base_save_path"]

IMAGENET_SPLIT = 'val'  # 'train' 'val'
CREATE_OOD = True
MAX_CLASSES = 0
IMGS_PER_CLASS = 0
EMBEDDING_MODE = 'resnet'  # 'clip' +'resnet'

METHOD = "naive_clip_definition"  # "hierarchy_definition" "naive_clip_definition"

DATASET = "spiders"  # "birds" "garments" "spiders" "ungulate" "fruit" "aircraft"


if DATASET == "garments":
    ood_synset = wn.synset('furniture.n.01')
    ind_synset = wn.synset('garment.n.01')
    seek_syn = wn.synset("garment.n.01")
    seek_word = "garment"
    ds_names = ["garments", "furniture"]
elif DATASET == "birds":
    ood_synset = wn.synset('hunting_dog.n.01') # wn.synset('hunting_dog.n.01')  wn.synset('furniture.n.01')
    ind_synset = wn.synset('bird.n.01') # wn.synset('bird.n.01')  wn.synset('garment.n.01')
    seek_syn = wn.synset("bird.n.01")
    seek_word = "bird"
    ds_names = ["birds", "huntingdog"]
elif DATASET == "spiders":
    ood_synset = wn.synset('butterfly.n.01')
    ind_synset = wn.synset('spider.n.01')
    seek_syn = wn.synset("spider.n.01")
    seek_word = "spider"
    ds_names = ["spiders", "butterflies"]
elif DATASET == "ungulate":
    ood_synset = wn.synset('bear.n.01')
    ind_synset = wn.synset('ungulate.n.01')
    seek_syn = wn.synset("ungulate.n.01")
    seek_word = "ungulate"
    ds_names = ["ungulate", "bear"]
elif DATASET == "fruit":
    ood_synset = wn.synset('musical_instrument.n.01')
    ind_synset = wn.synset('edible_fruit.n.01')
    seek_syn = wn.synset("edible_fruit.n.01")
    seek_word = "fruit"
    ds_names = ["fruit", "musical_instrument"]
elif DATASET == "aircraft":
    ood_synset = wn.synset('vessel.n.02')
    ind_synset = wn.synset('aircraft.n.01')
    seek_syn = wn.synset("aircraft.n.01")
    seek_word = "aircraft"
    ds_names = ["aircraft", "vessel"]
else:
    raise ValueError("Invalid dataset")

if CREATE_OOD:
    target_synset = ood_synset
else:
    target_synset = ind_synset

print(target_synset.definition())

DATASET_SAVE_PATH = f"{BASE_SAVE_PATH}datasets/imagenet_{DATASET}/imagenet_{ds_names[1] if CREATE_OOD else ds_names[0]}_allclasses_all{('val' if IMAGENET_SPLIT == 'val' else '')}im_{EMBEDDING_MODE}nofilter"
print(f"saving to: {DATASET_SAVE_PATH}")

# Create directory
path = f"{BASE_SAVE_PATH}datasets/imagenet_{DATASET}"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
   
from glob import glob
if IMAGENET_SPLIT == 'train':
    imagenet_path = IMAGENET_TRAIN_PATH
elif IMAGENET_SPLIT == 'val':
    imagenet_path = IMAGENET_VAL_PATH
else:
    raise ValueError("Invalid imagenet split")

class_folders = glob(imagenet_path+"/*/")
class_folders.sort()

folder_names = [x.split('/')[-2] for x in class_folders]
pos_and_offsets = [(s[0], int(s[1:])) for s in folder_names]

def has_path_to(source_synset, target_synset):
    open_list = [source_synset]
    closed_list = [target_synset]
    while open_list:
        current_node = open_list.pop()
        if current_node == target_synset:
            return True
        if current_node in closed_list:
            continue
        closed_list.append(current_node)
        open_list = open_list + current_node.hypernyms()
    return False

i=0
idx = 0
filtered_folders = []
filtered_names = []
for pos, offset in pos_and_offsets:
    syn = wn.synset_from_pos_and_offset(pos,offset)
    if has_path_to(syn, target_synset):
        filtered_folders.append(class_folders[idx])
        filtered_names.append(class_folders[idx].split('/')[-2])
        i+=1
    idx += 1
print(f"there are {i} classes in the dataset")

# Loading models
clip_model = clip.load("ViT-B/32")
embedding = Embedding(clip_model)
emb_dict = [seek_word, 'other']
word_model = WordModel(embedding, emb_dict)
emb_dict = DICT_FILE_PATH
concept_model = ConceptModel(embedding, emb_dict)
resnet_model = models.resnet50(pretrained=True).cuda()
resnet_model.eval()

def load_clip(img, return_tensor=False):
    img = Image.open(img)
    in_img = clip_model[1](img)
    with torch.no_grad():
        out = clip_model[0].encode_image(in_img.unsqueeze(0).cuda())
        out /= out.norm(dim=-1, keepdim=True)
    if return_tensor:
        return out
    else:
        return out.cpu().numpy().astype(np.float32)

def text_clip(tokenized):
    with torch.no_grad():
        text_features = clip_model[0].encode_text(tokenized)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def load_resnet(img):
    img = Image.open(img).convert("RGB")
    img = F.resize(img, [256], interpolation=InterpolationMode.BILINEAR)
    img = F.center_crop(img, [224])
    if not isinstance(img, torch.Tensor):
        img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    loaded_img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).cuda()
    # loaded_img = embedding.load_image(img, max_size=400, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()
    with torch.no_grad():
        out = resnet_model(loaded_img)
    return torch.nn.functional.normalize(out.float(), p=2.0, dim=-1).cpu().numpy().reshape(1,1000)

# Create dataset
if EMBEDDING_MODE == 'resnet':
    DIMS = 1000
else:
    DIMS = 512

db = px.Writer(dirpath=DATASET_SAVE_PATH, map_size_limit=500000, ram_gb_limit=16)

fid=0
for folder in filtered_folders:
    print(f"Embeddings Class {fid}")
    imgs = glob(folder+"*.JPEG")
    imgs.sort()
    num_data = min(len(imgs), IMGS_PER_CLASS if IMGS_PER_CLASS > 0 else len(imgs))
    data = np.empty([num_data, DIMS])
    if CREATE_OOD:
        labels = np.ones([num_data, 1])*(-1)
    else:
        labels = np.ones([num_data, 1])*fid
    imid=0
    for img in tqdm(imgs):
        if EMBEDDING_MODE == 'clip':
            data[imid,:] = load_clip(img)
        elif EMBEDDING_MODE == 'resnet':
            data[imid,:] = load_resnet(img)
        else:
            raise ValueError("invalid embedding mode")
        imid += 1
        if imid >= IMGS_PER_CLASS > 0:
            break
    db.put_samples('X', data, 'y', labels)
    fid += 1
    if fid >= MAX_CLASSES > 0:
        break
db.close()

if EMBEDDING_MODE == "resnet":
    exit()

# Compute OOD scores
scores = []

if METHOD == "hierarchy_definition":
    h_tree = wordnet_utils.find_tree_leaves(concept_model.wn_hierarchy, seek_syn)  # Only leaves
    text_input = torch.cat([clip.tokenize(syn.definition()) for syn in h_tree]).cuda()
    seek_text_features = text_clip(text_input)
    method_name_mnemonic = "seekdefinition"
elif METHOD == "naive_clip_definition":
    text_input = torch.cat([clip.tokenize(seek_syn.definition())]).cuda()
    seek_text_features = text_clip(text_input)
    method_name_mnemonic = "naivedefinition"
elif METHOD == "hierarchy_lemmas":
    h_tree = wordnet_utils.get_subtree_set(concept_model.wn_hierarchy, seek_syn)
    lemmas = [clip.tokenize(l.name().replace("_", " ")) for syn in h_tree for l in syn.lemmas()]
    text_input = torch.cat(lemmas).cuda()
    seek_text_features = text_clip(text_input)
    method_name_mnemonic = "seeklemmas"
else:
    raise ValueError("Invalid METHOD")

tot_id = 0
fid=0
for folder in filtered_folders:
    print(f"OOD Scores Class {fid}")
    imgs = glob(folder+"*.JPEG")
    imgs.sort()
    im_id = 0
    for img in tqdm(imgs):
        v = load_clip(img)
        if METHOD == "closest_syn_disambiguate":
            v_tree = concept_model.vector_get_synset_tree(v)
            try:
                leaves = wordnet_utils.find_tree_leaves(v_tree, target_synset)
                text_input = torch.cat([clip.tokenize(leaf.definition()) for leaf in leaves]).cuda()
                text_features = text_clip(text_input)
                similarities = torch.tensordot(torch.from_numpy(v).cuda().half(),text_features,dims=([1], [1])).squeeze()
                v_syn = leaves[torch.argmax(similarities)]
            except nx.NetworkXError:
                leaves = []
                v_syn = None

            if v_syn is not None and has_path_to(v_syn, seek_syn):
                scores.append((1,1))
            else:
                scores.append((0,1))
                nx.draw_networkx(v_tree)
                plt.show()
                Image.open(img).show()
                print(tot_id)
        elif METHOD == "hierarchy_lemmas":
            similarity = torch.max(torch.tensordot(torch.from_numpy(v).cuda().half(),seek_text_features,dims=([1], [1])).squeeze()).cpu().numpy()
            scores.append((similarity,1))
        elif METHOD == "hierarchy_definition":
            similarity = torch.max(torch.tensordot(torch.from_numpy(v).cuda().half(),seek_text_features,dims=([1], [1])).squeeze()).cpu().numpy()
            scores.append((similarity,1))
        elif METHOD == "naive_clip_definition":
            similarity = torch.max(torch.tensordot(torch.from_numpy(v).cuda().half(),seek_text_features,dims=([1], [1])).squeeze()).cpu().numpy()
            scores.append((similarity,1))
        else:
            raise ValueError("Invalid method")

        tot_id += 1
        im_id += 1
        if im_id >= IMGS_PER_CLASS > 0:
            break
    fid += 1
    if fid >= MAX_CLASSES > 0:
        break

SCORES_SAVE_PATH = f"{BASE_SAVE_PATH}ood_scores/scores_{ds_names[1] if CREATE_OOD else ds_names[0]}_allclasses_all{('val' if IMAGENET_SPLIT == 'val' else '')}im_{EMBEDDING_MODE}{method_name_mnemonic}"
print(f"saving to: {SCORES_SAVE_PATH}")

import pickle
with open(SCORES_SAVE_PATH, "wb") as f:
    pickle.dump(scores, f)