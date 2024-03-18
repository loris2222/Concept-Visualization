import pickle
import numpy as np
from generate_heatmap import *
from tqdm import tqdm
from utils import read_paths

paths_data = read_paths("./release/paths.json")
CONCEPT = "bird.n.01"
DATASET_BASE_PATH = paths_data["cub_base_path"]
N_IMAGES = 1000

# eval functions
from PIL import ImageDraw

def load_dataset():
    permute = True

    bbox_path = DATASET_BASE_PATH+"bounding_boxes.txt"
    images_path = DATASET_BASE_PATH+"images.txt"
    base_path = DATASET_BASE_PATH+"images/"

    with open(bbox_path, 'r') as file:
        bbox_list = [x.strip() for x in file.readlines()]

    with open(images_path, 'r') as file:
        images_list = [x.strip() for x in file.readlines()]

    assert len(bbox_list) == len(images_list)

    if permute:
        perm = np.random.permutation(len(images_list)).astype(int)
        bbox_list = [bbox_list[e] for e in perm]
        images_list = [images_list[e] for e in perm]

    return bbox_list, images_list

def get_path_from_id(image_id):
    path = DATASET_BASE_PATH + "images/" + images_list[image_id].split(" ")[1]
    return path

def get_bbox_from_id(image_id):
    line = bbox_list[image_id].split(" ")
    bbox = [float(line[i]) for i in range(1,5)]
    return bbox

def correct_bbox(bbox, img_orig_size):
    new_bbox = [0,0,0,0]
    w,h = img_orig_size
    mins = min(w,h)
    crop_x, crop_y = (0,0)
    if w > h:
        crop_x = w-h
    if h  > w:
        crop_y = h-w

    new_bbox[0] = bbox[0] - math.floor(crop_x/2)
    new_bbox[2] = bbox[2] + (new_bbox[0] if new_bbox[0]<0 else 0)  # If I have cut part of the bbox, the new width will be reduced
    new_bbox[0] = max(0, new_bbox[0])  # Clamp to valid region
    new_bbox[2] = min(mins-new_bbox[0], new_bbox[2])

    new_bbox[1] = bbox[1] - math.floor(crop_y/2)
    new_bbox[3] = bbox[3] + (new_bbox[1] if new_bbox[1]<0 else 0)  # If I have cut part of the bbox, the new width will be reduced
    new_bbox[1] = max(0, new_bbox[1])  # Clamp to valid region
    new_bbox[3] = min(mins-new_bbox[1], new_bbox[3])

    ratio = 224 / mins
    new_bbox = [int(x * ratio) for x in new_bbox]

    return new_bbox

def bbox_to_xys(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def bb_intersection_over_union(boxA, boxB):
    xya = bbox_to_xys(boxA)
    xyb = bbox_to_xys(boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(xya[0], xyb[0])
    yA = max(xya[1], xyb[1])
    xB = min(xya[2], xyb[2])
    yB = min(xya[3], xyb[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((xya[2] - xya[0]) * (xya[3] - xya[1]))
    boxBArea = abs((xyb[2] - xyb[0]) * (xyb[3] - xyb[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def eval_image(image_id, bbox_func, verbose=False, bbox_func_args=()):
    img_path = get_path_from_id(image_id)
    orig_size = load_opt_image(img_path)
    gt_bbox = get_bbox_from_id(image_id)
    gt_bbox = correct_bbox(gt_bbox, orig_size)
    pred_bbox = bbox_func(*bbox_func_args)
    if verbose:
        print("Predicted:")
        print(pred_bbox)
    if not isinstance(pred_bbox, list):
        pred_bbox = [pred_bbox]
    orig_size = load_opt_image(img_path)  # TODO remove, this is needed in multiscale
    max_iou = 0
    best_bbox = -1
    idx = 0
    for elem in pred_bbox:
        iou = bb_intersection_over_union(gt_bbox, elem)
        if iou > max_iou:
            max_iou = iou
            best_bbox = idx
        idx += 1
    # Prints
    if verbose:
        print(max_iou)
        viz_im = pil_im.copy()
        draw = ImageDraw.Draw(viz_im)
        draw.rectangle(bbox_to_xys(pred_bbox[best_bbox]), outline=(255, 0, 0), width=2)
        draw.rectangle(bbox_to_xys(gt_bbox), outline=(0, 255, 0), width=2)
        viz_im.show()
    return max_iou

def bbox_from_map(image_id, thresh):
    map = np.copy(maps[image_id])
    map[map<thresh] = 0
    map[map>=thresh] = 1

    normalized = getLargestCC(map)

    xys = np_bbox2(normalized)
    ratio = 224/normalized.shape[0]
    xys = [int((x+0.5)*ratio) for x in xys]

    # xys are rmin, rmax, cmin, cmax
    return xys[2], xys[0], xys[3]-xys[2], xys[1]-xys[0]

def eval_at_t(image_id, thresh):
    img_path = data[image_id][0]
    orig_size = load_opt_image(img_path)
    gt_bbox = data[image_id][1]
    gt_bbox = correct_bbox(gt_bbox, orig_size)
    pred_bbox = bbox_from_map(image_id, thresh)

    if not isinstance(pred_bbox, list):
        pred_bbox = [pred_bbox]

    max_iou = 0
    best_bbox = -1
    idx = 0
    for elem in pred_bbox:
        iou = bb_intersection_over_union(gt_bbox, elem)
        if iou > max_iou:
            max_iou = iou
            best_bbox = idx
        idx += 1

    return max_iou

def generate_maps(params, n_test):
    result = np.zeros([n_test,224,224])
    for image_id in tqdm(range(n_test)):

        map = generate_map(get_path_from_id(image_id), CONCEPT, params)
        result[image_id] = map
    return result

def get_image_data(n_im):
    names = []
    bboxes = []
    for image_id in range(n_im):
        names.append(get_path_from_id(image_id))
        bboxes.append(get_bbox_from_id(image_id))
    res = list(zip(names, bboxes))
    return res


if __name__ == "__main__":
    params = {'eps': 0.03, 'iters': 500, 'rank_gamma': 4, 'restart_each': 100, 'thresh': 0.7}
    bbox_list, images_list = load_dataset()

    maps = generate_maps(params, N_IMAGES)
    data = get_image_data(N_IMAGES)
    
    # MaxBoxAcc
    ress = []
    for t in tqdm(list(np.arange(0, 1, 0.01))):
        # print(f"t:{t}: ", end="")
        ok = 0
        for i in range(N_IMAGES):
            r = eval_at_t(i,t)
            if r > 0.5:
                ok += 1
        # print(ok)
        ress.append(ok/N_IMAGES)
    print(f"MaxBoxAcc: {max(ress)}")
    