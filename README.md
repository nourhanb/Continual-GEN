# Continual-GEN
This is the PyTorch implementation of our ISICW MICCAI 2023 paper (**Continual-GEN: Continual Group Ensembling for Domain-agnostic Skin Lesion Classification**).

![](overview.png)

## File description
- baseline_training: Training algorithm of the cross-entropy baseline model.
- constrastive_training: Includes both SupCon and SimCLR trainings, the desired one can be selected by changing --method argument.
- contrastive_test: Evaluation of the contrastive models, calculating their global separation by class and k-means clusters.
- losses: Defines the constrastive loss function.
- util: Contains functions used throughout the project.
- prompts: Command promts to execute each training file. The examples inside are those used to train the final models.

## Requirements
All the necessary libraries are listed in the requirements.txt file. You will also need:
- Python == 3.10
- CUDA == 11.7
- Cudnn == 8.5
- Pytorch == 2.0.0

## Training and Evaluation
Here you can find the command prompts used to run each training method. You can edit the hyperparameters here or inside the corresponding code file.

- Cross-Entropy baseline: `python baseline_training.py --batch_size 16 --learning_rate 0.001 --epochs 150 --trial 2`
- SupCon: `python contrastive_training.py --batch_size 16 --learning_rate 0.001 --epochs 150 --temp 0.1 --trial 0`
- SimCLR: `python contrastive_training.py --batch_size 16 --learning_rate 0.001 --epochs 150 --method SimCLR --temp 0.1 --trial 0`

Testing contrastive models
- `python contrastive_test.py --method CE`
- `python contrastive_test.py --method SupCon`
- `python contrastive_test.py --method SimCLR`
  
#In method you choose between SimCLR or SupCon

## References
https://github.com/HobbitLong/SupContrast

## Citation
If you use this code in your research, please consider citing:
```bash
@inproceedings{bayasi2023continual-gen,
title={{Continual-GEN}: Continual Group Ensembling for Domain-agnostic Skin Lesion Classification},
author={Bayasi, Nourhan and Du, Siyi and Hamarneh, Ghassan and Garbi, Rafeef},
booktitle={26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2023) ISIC Workshop}}


