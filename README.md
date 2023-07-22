# ood-detection
Out of distribution data points detection with contrastive learning and clustering in Python

## Folder description
- data: Contains the HAM dataset
- ham: Class and functions for creating data loaders from HAM dataset
- networks: Includes different resnet architectures to use, as well as Supcon and SimCLR models.
- save: Saved models and history records after each model training.
- test: A couple of notebooks for testing each part of the training files during the project development. You could use it or ignore it.

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
For this you could create a conda enviroment and install the dependencies in the requirements file first.
Then go to this page (https://pytorch.org/get-started/locally/) and get the command to put in the anaconda prompt to install cuda.

Concerning about the data, you must create a "save" folder and download the HAM dataset from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
Decompress the files and merge all skin images in a single "images" folder inside "save".

## References
https://github.com/HobbitLong/SupContrast
https://arxiv.org/pdf/2203.08549.pdf
https://arxiv.org/pdf/2004.11362.pdf
