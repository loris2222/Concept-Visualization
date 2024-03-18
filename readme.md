# How to use

  

## Initial setup

Setup the paths in `paths.json`

 - Base save path should point to the relative path that contains the root of the codebase.
 - Imagenet train path should point to the imagenet training data folder
 - Original imagenet val should point to the imagenet validation folder
 - Imagenet val path should point to a folder containing the imagenet validation images organised in class folders. Use `prepare_imagenet_val.py` to do so.
 - Dict file path should point to the dictionary of concepts that make up T. It is pre-compiled
 - Cub base path should point to your CUB dataset folder
 - COCO base path should point to your COCO dataset folder
 - Maps save path is the path where to save the saliency maps to be used for the webapp

## Generating a single saliency map

Function `generate_map(img_path, concept)` in `generate_heatmap.py`. First argument is a path to the image for which to generate the saliency map, second argument is the synset concept name.

## Evaluating OOD detection

### Prepare the pre-processed features and scores
Run `build_ood_scores.py` for each dataset split, in/out distribution, and scoring methodology:
- train, ind, resnet
- val, ood, resnet
- val, ind, resnet
- val, ood, clip, naive_clip
- val, ind, clip, naive_clip
- val, ood, clip, hierarchy_clip
- val, ind, clip, hierarchy_clip

By modifying the parameters of the script.
You should be left with the pre-computed scores and datasets in the folders `/release/ood_scores`
and `/release/datasets`.

### Train classifiers
Run `train_resnet_classifier.py` for each of the datasets by modifying the parameters of the script.

You should be left with a series of checkpoints in `/release/model_weights`.

### Evaluate
Run `evaluate_ood.py` for all datasets.

## Evaluating WSOL

Run `wsol_evaluation.py`

## WebApp

### Prepare data
Run `/webapp_utils/create_maps.py` by setting `image_id` for all required samples.

You should be left with all preprocessed maps in the selected save path

Run `/webapp_utils/get_pixel_counts.py` by setting the path to the maps to compute the sorting weigths.

### Run
Setup a server of your choice and run the React App.

Optionally setup a database to store results