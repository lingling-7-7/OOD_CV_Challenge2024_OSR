This repo contains the code for [OOD-CV Workshop SSB Challenge 2024 (Open-Set Recognition Track)](https://codalab.lisn.upsaclay.fr/competitions/19341?secret_key=25c47dd3-065f-41bc-ae40-6ba6f737ff32#learn_the_details)

# Setup
Please refer [SSB](https://github.com/sgvaze/SSB) to setup,including: setting up a Kaggle account;
setting up an SSB JSON config; installing python requirements; and installing this SSB Python package. 

# Download datasets

After setup, please refer the download part in [SSB](https://github.com/sgvaze/SSB). This is an example:
```
from SSB.download import download_datasets
download_datasets(['cub', 'aircraft', 'scars', 'imagenet_1k', 'imagenet_21k'])
```


# Get Open-Set Recognition (OSR) datasets

Only ImageNet is currently supported in the OSR challenge. 
The SSB split (i.e 'Easy' or 'Hard') should be specified in ```osr_split```.

Documentation is given inside the function in ```SSB/get_datasets/get_osr_datasets_funcs.py```.

Specifying ```eval_only=True``` means only the test datasets will be returned (the ImageNet-1k training set is not loaded). This is faster.

```
from SSB import get_osr_datasets
all_datasets = get_osr_datasets(dataset_name='imagenet',
                                osr_split='Hard', 
                                train_transform=None, 
                                test_transform=test_transform,
                                eval_only=True)
```

# Our solution
Our approach is improved based on BASELINE, including the following two points:

- Multi-model joint inference for pre-trained weights of the same image size
- Multi-model joint inference based on data augmentation

# Usage

If you want to use joint inference only, you can 
```
python OOD-OSR/evaluate_osr_model_fusion.py
```
If you want to add data augment in joint inference, you can
```
python OOD-OSR/evaluate_osr_TTA.py
```

# Citation

```
@InProceedings{vaze2022openset,
      title={Open-Set Recognition: a Good Closed-Set Classifier is All You Need?},
      author={Sagar Vaze and Kai Han and Andrea Vedaldi and Andrew Zisserman},
      booktitle={International Conference on Learning Representations},
      year={2022}
      }
```
