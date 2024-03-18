import numpy as np
import os
from PIL import Image
import json

BASE_PATH = "./"

directories = [name for name in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, name))]

result = {}

for dir in directories:
	dir_res = {}
	files = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
	for file in files:
		file = file.replace("/", "_")
		imarray = np.array(Image.open(BASE_PATH + dir + "/" + file))
		sum = np.sum(imarray)
		dir_res[file] = int(sum)
	result[dir] = dir_res
	print(dir)

with open("counts.json", "w") as file:
	json.dump(result, file)