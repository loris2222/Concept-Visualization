from utils import read_paths
from tqdm import tqdm
import os
import shutil

paths_data = read_paths("./release/paths.json")

BASE_SAVE_PATH = paths_data["base_save_path"]

SCRIPT_PATH = BASE_SAVE_PATH + "scripts/valprep.sh"
IMAGENET_VAL_PATH = paths_data["original_imagenet_val"]
TARGET_PATH = paths_data["imagenet_val_path"]

with open(SCRIPT_PATH, 'r') as f:
    lines = f.readlines()


with tqdm(total=100) as pbar:
    l = len(lines)
    i = 0
    for line in lines:
        spl = line.split(" ")
        cmd = spl[0]
        if cmd == "mkdir":
            os.mkdir(os.path.join(TARGET_PATH, spl[2].strip()))
        elif cmd == "mv":
            src = os.path.join(IMAGENET_VAL_PATH, spl[1].strip())
            dst = os.path.join(TARGET_PATH, spl[2].strip()) + spl[1].strip()
            shutil.copyfile(src, dst)
        else:
            raise ValueError("Invalid command")
        i += 1
        pbar.n = (i/l)*100
        pbar.refresh()