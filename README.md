# Continual-GEN
This is the PyTorch implementation of our ISICW MICCAI 2023 paper (**Continual-GEN: Continual Group Ensembling for Domain-agnostic Skin Lesion Classification**).

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

## References
https://github.com/HobbitLong/SupContrast
