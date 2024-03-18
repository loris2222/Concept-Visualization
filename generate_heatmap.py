from matplotlib import pyplot as plt
from language_utils import ConceptModel
import clip
from utils import Embedding, read_paths
import torch
from PIL import Image
import numpy as np
import wordnet_utils
import math
from PIL import ImageFilter, ImageEnhance
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
import cv2
from skimage.measure import label
from nltk.corpus import wordnet as wn
import os
import requests
import tempfile
import random
import string

paths_data = read_paths("./release/paths.json")

CONCEPT_DICT_PATH = paths_data["dict_file_path"]
SEEK_TYPE = "definitions"  # "definitions" "lemmas"

# Loading clip model and language hierarchy
clip_model = clip.load("ViT-B/32")
embedding = Embedding(clip_model)
emb_dict = CONCEPT_DICT_PATH
concept_model = ConceptModel(embedding, emb_dict)

def text_clip(tokenized):
    with torch.no_grad():
        text_features = clip_model[0].encode_text(tokenized)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def pil_centercrop(pil_im):
    w, h = (pil_im.width, pil_im.height)
    crop_x, crop_y = (0,0)
    if w > h:
        crop_x = w-h
    if h  > w:
        crop_y = h-w
    return pil_im.crop((math.floor(crop_x/2), math.floor(crop_y/2), w-math.ceil(crop_x/2), h-math.ceil(crop_y/2)))

def pil_resize(pil_im):
    return pil_im.resize((224,224))

def np_bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def load_clip_mask_filtered(img, mask, show=False, crop_to_mask=True, brightness=0.5, blur_sigma=1, noise_image=False):
    """
    Takes a pil img and a binary mask of the same size and returns the clip
    embedding of the image blurred where the mask is 0
    :param img:
    :param mask:
    :return:
    """
    try:
        mask_bbox = np_bbox2(mask)
    except IndexError:
        mask_bbox = None

    img = img.copy()
    mask = np.tile(mask[:,:,np.newaxis], (1,1,4)).astype(float)
    mask_im = Image.fromarray(((1.0-mask)*255).astype(np.uint8)).convert("L")
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(radius=0.5))

    if noise_image:
        blurred = Image.fromarray(np.random.randint(0,255,(img.height,img.width,3),dtype=np.dtype(np.uint8)))
    else:
        blurred = img.copy()

    br_filter = ImageEnhance.Brightness(blurred)
    blurred = br_filter.enhance(brightness)
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=blur_sigma))


    img.paste(blurred, (0,0), mask=mask_im)

    if mask_bbox is not None and crop_to_mask:
        img = img.crop((mask_bbox[2], mask_bbox[0], mask_bbox[3], mask_bbox[1]))

    if show:
        img.show()

    in_img = clip_model[1](img)
    with torch.no_grad():
        out = clip_model[0].encode_image(in_img.unsqueeze(0).cuda())
    return torch.nn.functional.normalize(out.float(), p=2.0, dim=-1).cpu().numpy().reshape(1,512)


def load_clip_box_filtered(img, bbox):
    # TODO don't use embedding, just use the pil image img and the bbox to construct the blurred version
    loaded_img, pil_im = embedding.load_image(img, max_size=400, return_pilimage=True)
    img_emb = embedding.get_image_embedding(loaded_img)
    mask_np = np.zeros(img_emb.shape[1:3])
    mask_np[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
    mask = np.tile(mask_np[:,:,np.newaxis], (1,1,4)).astype(float)
    mask = np.pad(mask, ((0,int(pil_im.height/2)-mask.shape[0]),(0,int(pil_im.width/2)-mask.shape[1]), (0,0)), mode='constant', constant_values=0)
    mask_im = Image.fromarray(((1.0-mask)*255).astype(np.uint8)).convert("L")
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(radius=0.5))

    pil_im = pil_im.resize((int(pil_im.width/2), int(pil_im.height/2)))
    blurred = pil_im.copy()
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=20))
    pil_im.paste(blurred, (0,0), mask=mask_im)

    in_img = clip_model[1](pil_im)
    with torch.no_grad():
        out = clip_model[0].encode_image(in_img.unsqueeze(0).cuda())
    return torch.nn.functional.normalize(out.float(), p=2.0, dim=-1).cpu().numpy().reshape(1,512)


def load_clip_box_cropped(img, bbox):
    pil_im = img.crop(bbox)

    in_img = clip_model[1](pil_im)

    with torch.no_grad():
        out = clip_model[0].encode_image(in_img.unsqueeze(0).cuda())
    return torch.nn.functional.normalize(out.float(), p=2.0, dim=-1).cpu().numpy().reshape(1,512)

def load_opt_image(path):
    global IMG_PATH, pil_im
    IMG_PATH = path 
    pil_im = Image.open(IMG_PATH).convert("RGB")
    orig_size = pil_im.size
    pil_im = pil_resize(pil_centercrop(pil_im))
    return orig_size

def pad_opt_image(pad_amt, noise_image=False):
    global pil_im
    if noise_image:
        new_im = Image.fromarray(np.random.randint(0,255,(pil_im.height+2*pad_amt,pil_im.width+2*pad_amt, 3),dtype=np.dtype(np.uint8)))
    else:
        new_im = Image.new("RGB", (pil_im.width+2*pad_amt,pil_im.height+2*pad_amt))
    new_im.paste(pil_im, (pad_amt, pad_amt))
    pil_im = new_im

def project_params(params, min_size=0):
    params[0] = np.clip(params[0], 0, pil_im.width - min_size)
    params[1] = np.clip(params[1], 0, pil_im.height - min_size)
    params[2] = np.clip(params[2], min_size, pil_im.width-params[0])
    params[3] = np.clip(params[3], min_size, pil_im.height-params[1])
    return params


def raw_similarity(params, verbose=False, crop_mask=False, invert_mask=False, brightness=0.5, blur_sigma=1, noise_image=False, aggregate=True):
    """
    Computes loss given the parameters of the bbox
    :param params: bbox as [left, top, width, height]
    :return: similarity with concept
    """
    mask = np.zeros((pil_im.height, pil_im.width))
    mask[params[1]:params[1]+params[3], params[0]:params[0]+params[2]] = 1

    if invert_mask:
        mask = np.logical_not(mask)

    v = load_clip_mask_filtered(pil_im, mask, show=verbose, crop_to_mask=crop_mask, brightness=brightness, blur_sigma=blur_sigma, noise_image=noise_image)

    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    similarities = torch.tensordot(torch.from_numpy(v).cuda().half(), seek_text_features, dims=([1], [1])).squeeze()

    #Debug
    if verbose:
        sorted_all_idx = torch.flip(torch.argsort(similarities), dims=[0])
        remapped_similarities = [(h_list[x], similarities[x].cpu().numpy()) for x in sorted_all_idx[0:20]]
        print("Top k")
        print(remapped_similarities)
        remapped_similarities = [h_list[x] for x in sorted_all_idx[0:20] if x in leaves_idxs]
        print("Of which in seek list")
        print(remapped_similarities)

    if aggregate:
        similarities = similarities[leaves_idxs]

        idx_sorted = torch.flip(torch.argsort(similarities), dims=[0])
        similarities = similarities[idx_sorted]
        concept_similarity = np.max(similarities.cpu().numpy(), axis=0)
        return concept_similarity
    else:
        return similarities

def binary_similarity(params, invert_mask=False, brightness=0.5, blur_sigma=1, verbose=False, noise_image=False, top_k=1):
    """
    returns 1 if top definition is in seek, else 0
    :param params:
    :return:
    """
    mask = np.zeros((pil_im.height, pil_im.width))
    mask[params[1]:params[1]+params[3], params[0]:params[0]+params[2]] = 1

    if invert_mask:
        mask = np.logical_not(mask)


    v = load_clip_mask_filtered(pil_im, mask, show=verbose, crop_to_mask=False, brightness=brightness, blur_sigma=blur_sigma, noise_image=noise_image)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    similarities = torch.tensordot(torch.from_numpy(v).cuda().half(), seek_text_features, dims=([1], [1])).squeeze()
    sorted_all_idx = torch.flip(torch.argsort(similarities), dims=[0])

    if verbose:
        remapped_similarities = [(h_list[x], similarities[x].cpu().numpy()) for x in sorted_all_idx[0:2*top_k]]
        print("Top k")
        print(remapped_similarities)

    return any(elem in leaves_idxs_set for elem in sorted_all_idx[0:top_k].tolist())


def compute_similarity_bounds():
    max_sim = raw_similarity([0, 0, pil_im.width, pil_im.height])
    min_sim = raw_similarity([0, 0, 0, 0])
    # assert min_sim < max_sim  # TODO revert to this
    return min_sim, max_sim

def compute_loss(params, sim_bounds, alpha, return_raw=False, crop_mask=False, invert_mask=False, brightness=0.5, blur_sigma=1, noise_image=False, sim_good=True, area_good=False):
    params = project_params(params)

    box_area = (params[2]*params[3])
    area_ratio = box_area / (pil_im.height*pil_im.width)
    sim = raw_similarity(params, crop_mask=crop_mask, invert_mask=invert_mask, brightness=brightness, blur_sigma=blur_sigma, noise_image=noise_image)  # np.clip(raw_similarity(params), sim_bounds[0], sim_bounds[1])
    sim_ratio = (sim - sim_bounds[0]) / (sim_bounds[1]-sim_bounds[0])
    loss = alpha * ((1-area_ratio) if area_good else area_ratio) + (1-alpha) * ((1-sim_ratio) if sim_good else sim_ratio)
    if return_raw:
        return loss, sim, box_area
    else:
        return loss

def compute_rank(params, verbose=False, crop_mask=False, invert_mask=False, brightness=0.5, blur_sigma=1, noise_image=False, do_project_params=True):
    global leaves_idxs
    if do_project_params:
        params = project_params(params)

    similarities = raw_similarity(params, verbose=verbose, crop_mask=crop_mask, invert_mask=invert_mask, brightness=brightness, blur_sigma=blur_sigma, noise_image=noise_image, aggregate=False)

    argm = torch.argmax(similarities).cpu().numpy().astype(int)
    sorted_all_idx = torch.argsort(torch.flip(torch.argsort(similarities), dims=[0]))
    leaves_ranks = sorted_all_idx[leaves_idxs]
    return torch.min(leaves_ranks).cpu().numpy().astype(int)

def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def grid_heatmap():
    """
    This is the procedure according to algorithm 1
    """
    global pil_im

    box_sizes = [128, 64]
    stride = 16
    
    # Alternative to simple ranking function, you can exponentiate it to only consider concepts
    # that are at the top of the ranking. Disabled by default.
    USE_GAMMA = False
    rank_gamma = 5
    n_leaves = 91
    rank_cutoff = 91

    current_box_size = box_sizes[0]
    global_accumulator = None
    pad_opt_image(current_box_size, noise_image=True)
    accumulator = np.zeros([pil_im.height, pil_im.width])  # Initialise the results for all W_{l,m}

    # For patches at large scale
    for y in range(0,accumulator.shape[0], stride):
        for x in range(0,accumulator.shape[1], stride):
            # Compute r_{i,j}
            rank = compute_rank([x,y,current_box_size,current_box_size], verbose=False, crop_mask=True, invert_mask=False, brightness=0.0, blur_sigma=1, noise_image=True, do_project_params=False)
            if USE_GAMMA:
                exp_rank = 2 * ((np.exp(-rank_gamma*(rank/n_leaves)) if rank < rank_cutoff else 0) - 0.5)
                new_score = exp_rank
            else:
                new_score = rank

            # Add the r_{i,j} to all sets W_{l,m} that should contain it
            accumulator[y:y+current_box_size, x:x+current_box_size] += new_score

    # Now switch to patches at small scale
    current_box_size = box_sizes[1]
    for y in range(0,accumulator.shape[0], stride):
        for x in range(0,accumulator.shape[1], stride):
            rank = compute_rank([x,y,current_box_size,current_box_size], verbose=False, crop_mask=True, invert_mask=False, brightness=0.0, blur_sigma=1, noise_image=True, do_project_params=False)
            if USE_GAMMA:
                exp_rank = 2 * ((np.exp(-rank_gamma*(rank/n_leaves)) if rank < rank_cutoff else 0) - 0.5)
                new_score = exp_rank
            else:
                new_score = rank

            # Add the r_{i,j} to all sets W_{l,m} that should contain it
            accumulator[y:y+current_box_size, x:x+current_box_size] += new_score

    # Normalising here is equivalent to taking the mean over the scores in W and among the large and 
    # small patch size since all rank scores are already normalised.
    normalized = accumulator[int(box_sizes[0]):-int(box_sizes[0]), int(box_sizes[0]):-int(box_sizes[0])]
    normalized = (normalized-np.min(normalized))/(np.max(normalized)-np.min(normalized))
    if not USE_GAMMA:
        normalized = 1-normalized

    raw_map = np.copy(normalized)
    return raw_map

def stochastic_grid(verbose=False, return_all=False, config=None, return_raw_map=False):
    """
    Alternative to the standard algorithm.
    Since the OpenAI CLIP implementation does not allow batching in inference,
    speeds up computation by running the similarity on a random sample position instead
    of checking all the possible locations. Then uses a procedure similar to adaboost to
    increase the probability of sampling close to patches where there was similarity to
    the requested concept.

    You can otherwise modify grid_heatmap to allow batching and use that instead.
    """
    global pil_im
    assert return_all == False

    box_size = 128
    best_score = 0
    best_xy = None
    thresh = config["thresh"] if config else 0.7
    rank_gamma = config["rank_gamma"] if config else 5
    n_leaves = 91  # 91
    rank_cutoff = n_leaves
    eps = config["eps"] if config else 0.002
    iters = int(config["iters"]) if config else 600
    restart_each = int(config["restart_each"]) if config else 100

    global_accumulator = None
    pad_opt_image(box_size, noise_image=True)
    accumulator = np.ones([pil_im.height, pil_im.width])

    current_box_size = box_size
    for j in range(iters):
        # Sample following distribution of accumulator on valid locations
        pdf = accumulator[0:224+box_size,0:224+box_size]
        pdf = pdf/np.sum(pdf)
        flat = pdf.flatten()
        sample_index = np.random.choice(a=flat.size, p=flat)
        y, x = np.unravel_index(sample_index, pdf.shape)  # These are on 0,0 reference frame on accumulator

        rank = compute_rank([x,y,current_box_size,current_box_size], verbose=False, crop_mask=True, invert_mask=False, brightness=0.0, blur_sigma=1, noise_image=True, do_project_params=False)
        exp_rank = 2 * ((np.exp(-rank_gamma*(rank/n_leaves)) if rank < rank_cutoff else 0) - 0.5)
        new_score = exp_rank

        accumulator[y:y+current_box_size, x:x+current_box_size] *= 1 + eps * new_score

        if j % restart_each == restart_each-1:
            if global_accumulator is None:
                global_accumulator = np.copy(accumulator)
            else:
                global_accumulator += accumulator
            accumulator = np.ones([pil_im.height, pil_im.width])
            if j >= iters/2 - 1:
                current_box_size = 64

    normalized = global_accumulator[box_size:-box_size, box_size:-box_size]
    normalized = (normalized-np.min(normalized))/(np.max(normalized)-np.min(normalized))
    if verbose:
        print(pdf)
        plt.imshow(normalized)
        plt.colorbar()
        plt.show()

    raw_map = np.copy(normalized)

    normalized[normalized<thresh] = 0
    normalized[normalized>=thresh] = 1

    normalized = getLargestCC(normalized)

    xys = np_bbox2(normalized)
    ratio = 224/normalized.shape[0]
    xys = [int((x+0.5)*ratio) for x in xys]

    if verbose:
        print(pdf)
        rescaled = (np.pad(np.kron(normalized*0.5+0.5, np.ones([int(ratio),int(ratio)])), [[0,0],[0,0]], constant_values=0)[:,:,np.newaxis] * np.array(pil_im.crop((box_size,box_size,224+box_size,224+box_size)))).astype(np.uint8)
        plt.imshow(rescaled)
        plt.colorbar()
        plt.show()
    # xys are rmin, rmax, cmin, cmax
    if return_raw_map:
        return raw_map
    else:
        return xys[2], xys[0], xys[3]-xys[2], xys[1]-xys[0]

# Generating language model
if SEEK_TYPE == "definitions":
    h_tree = wordnet_utils.get_subtree_set(concept_model.wn_hierarchy, concept_model.target_synset)
    h_list = list(h_tree)
    text_input = torch.cat([clip.tokenize(syn.definition()) for syn in h_tree]).cuda()
    seek_text_features = text_clip(text_input)
elif SEEK_TYPE == "lemmas":
    h_tree = wordnet_utils.get_subtree_set(concept_model.wn_hierarchy, concept_model.target_synset)
    h_list = [l.name().replace("_", " ") for syn in h_tree for l in syn.lemmas()]
    lemmas = [clip.tokenize(l.name().replace("_", " ")) for syn in h_tree for l in syn.lemmas()]
    text_input = torch.cat(lemmas).cuda()
    seek_text_features = text_clip(text_input)
else:
    raise ValueError("invalid seek type")

def download_image_and_save(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        # Generate a random filename
        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        
        # Get the file extension from the content type
        file_extension = response.headers.get('content-type').split('/')[-1]
        
        # Create a temporary file with the random name and extension
        temp_file_path = os.path.join(tempfile.gettempdir(), f"{random_filename}.{file_extension}")
        
        # Save the image to the temporary file
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(response.content)
        
        print(f"Saved temp image to {temp_file_path}")
        return temp_file_path
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
        return None


def generate_map(img_path, concept, method="stochastic", params=None):
    global pil_im, leaves, leaves_idxs, leaves_idxs_set, map_idx

    if img_path[0:4] == "http":
        img_path = download_image_and_save(img_path)

    # Creating list of concepts from tree
    map_synset = wn.synset(concept)
    leaves = wordnet_utils.get_subtree_set(concept_model.wn_hierarchy, map_synset)
    leaves = list(leaves)
    leaves_idxs = [h_list.index(x) for x in leaves]
    leaves_idxs_set = set(leaves_idxs)
    map_idx = h_list.index(map_synset)

    # Generate saliency map
    load_opt_image(img_path)
    if method == "grid":
        return grid_heatmap()

    if params==None:
        map_out = stochastic_grid(verbose=False, return_all=False, return_raw_map=True)
    else:
        map_out = stochastic_grid(False, False, params, True)
    return map_out

if __name__ == "__main__":
    global pil_im, leaves, leaves_idxs

    CONCEPT = "building.n.01"
    # IMG_PATH = r"./release/test_images/dogtigercat.png"
    # SAVE_PATH = r"./release/test_output/test.png"
    IMG_PATH = r"./release/test_images/coco2.jpg"
    SAVE_PATH = f"./release/test_output/web_{CONCEPT}.png"
    VIS_EXP = 10 # exponent to give to the heatmap to make it sharper. (def: 4 for stoch., 20 for grid)
    METHOD = "grid"  # "grid" is the standard from the algo, "stochastic" is a faster version

    map_custom = generate_map(IMG_PATH, CONCEPT, method=METHOD)

    # Visualize saliency map
    resized = pil_im.crop((128, 128, 352, 352))
    output_image = Image.new("RGB", (224, 224))

    for y in range(224):
        for x in range(224):
            saliency_value = map_custom[y, x]**VIS_EXP
            # saliency_value = min(0.2, saliency_value) if saliency_value<0.5 else saliency_value
            pixel_color = resized.getpixel((x, y))
            
            # Linear combination using saliency value
            r = int((1 - saliency_value) * 255 + saliency_value * pixel_color[0])
            g = int((1 - saliency_value) * 255 + saliency_value * pixel_color[1])
            b = int((1 - saliency_value) * 255 + saliency_value * pixel_color[2])
            
            output_image.putpixel((x, y), (r, g, b))
    # output_image.show()
    output_image.save(SAVE_PATH)