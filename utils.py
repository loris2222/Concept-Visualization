import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import clip
import json

class Embedding:
    def __init__(self, clip_model):
        self.words = ['other']
        self.text_embedding = None
        self.clip_model = clip_model

        torch.manual_seed(2222)
        alpha = 0.5

    @staticmethod
    def load_image(img_path, from_numpy=False, from_pil=False, size=None, max_size=None, mean=None, std=None, return_pilimage=False):
        assert not (from_numpy and from_pil)

        if from_numpy:
            image = Image.fromarray(img_path)
        elif from_pil:
            image = img_path.convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')

        if size is not None:
            image = image.resize(size)
        elif max_size is not None and (image.size[0] > max_size or image.size[1] > max_size):
            width, height = image.size
            factor = min(max_size/width, max_size/height)
            image = image.resize((int(width*factor), int(height*factor)))

        width, height = image.size

        cropleft, cropright, croptop, cropbottom = (0, 0, 0, 0)
        if width % 32 >= 16:
            cropx = width % 32 - 15
            cropleft = math.floor(cropx/2)
            cropright = math.ceil(cropx/2)
        if height % 32 >= 16:
            cropy = height % 32 - 15
            croptop = math.floor(cropy/2)
            cropbottom = math.ceil(cropy/2)

        image = image.crop((cropleft, croptop, width-cropright, height-cropbottom))
        pil_im = image

        image = np.array(image)

        if mean is None:
            mean = [0.5, 0.5, 0.5]
        if std is None:
            std = [0.5, 0.5, 0.5]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        image = transform(image).unsqueeze(0)
        if return_pilimage:
            return image, pil_im
        return image

    @staticmethod
    def show_image(image):
        img = image[0].permute(1, 2, 0)
        img = img * 0.5 + 0.5
        plt.imshow(img)
        plt.show()

    def load_txt_dict(self, txt_path):
        with open(txt_path, 'r') as f:
            words = f.readlines()
        words = [x.strip() for x in words]
        return words

    def set_dict(self, new_words):
        self.words = new_words
        self.text_embedding = self.get_text_embedding(self.words)

    def set_dict_emb(self, dict_emb):
        self.text_embedding = torch.from_numpy(dict_emb).cuda()
        
    def get_image_embedding(self, image):
        with torch.no_grad():
            return self.clip_model[0].encode_images(image)
    
    def get_text_embedding(self, words):
        with torch.no_grad():
            text_input = torch.cat([clip.tokenize(word) for word in words]).cuda()
            return self.clip_model[0].encode_text(text_input)

    def get_similarities(self, im_emb, text_emb=None):
        if text_emb is None:
            text_emb = self.text_embedding
            num_words = text_emb.shape[0]
        else:
            text_emb = torch.from_numpy(text_emb).cuda()
            num_words = text_emb.shape[0]  # TODO this was above, but I think here is the right spot

        sim_maps_cpu = np.zeros([im_emb.shape[1], im_emb.shape[2], num_words])

        DOT_BATCH_SIZE = 512
        last_size = num_words % DOT_BATCH_SIZE

        i = 0
        for i in range(np.floor(num_words / DOT_BATCH_SIZE).astype(np.uint8)):
            sim_maps_cpu[:, :, i * DOT_BATCH_SIZE:(i + 1) * DOT_BATCH_SIZE] = torch.tensordot(im_emb[0],
                                                                                              text_emb[i * DOT_BATCH_SIZE:(i + 1) * DOT_BATCH_SIZE],
                                                                                              dims=([2], [1])).cpu().numpy()
        sim_maps_cpu[:, :, i * DOT_BATCH_SIZE:num_words] = torch.tensordot(im_emb[0],
                                                                           text_emb[i * DOT_BATCH_SIZE:num_words],
                                                                           dims=([2], [1])).cpu().numpy()
        sim_maps_cpu = sim_maps_cpu.transpose((2, 0, 1))
        return sim_maps_cpu

def read_paths(paths_file):
    with open(paths_file) as f:
        data = json.load(f)
    
    return data