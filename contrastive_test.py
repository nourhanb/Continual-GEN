from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from util import AverageMeter, TwoCropTransform
from util import adjust_learning_rate, warmup_learning_rate, accuracy, calculate_global_separation, group_embeddings, apply_emb_clustering, calculate_cluster_purity, apply_gmm_clustering
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet,SupCEResNet
from losses import SupConLoss
from contrastive_training import load_data

from ham import create_dataframes, create_transformations, HAM10000

import warnings
# Filter out DeprecationWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ensure results are reproducible
np.random.seed(117)
torch.manual_seed(117)
torch.cuda.manual_seed(117)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR','CE'], help='choose method')
    
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '/ubc/ece/home/ra/grads/nourhanb/Documents/skins/DMF'
    opt.n_cls = 7
    opt.input_size = 100    # Size of the reshaped images (input_size, input_size)
    opt.batch_size = 16
    opt.num_workers = 8

    if opt.method == 'SupCon':
        opt.model_path = './save/SupCon/DMF_models/SupCon_DMF_resnet50_lr_0.001_decay_0.0001_bsz_16_temp_0.1_trial_0/last.pth'

    elif opt.method == 'SimCLR':
        opt.model_path = './save/SimCLR/DMF_models/SimCLR_DMF_resnet50_lr_0.001_decay_0.0001_bsz_16_temp_0.1_trial_0/last.pth'
    
    elif opt.method == 'CE':
        opt.model_path = './save/CE/DMF_models/SupCE_DMF_resnet50_lr_0.001_decay_0.0001_bsz_16_trial_2/last.pth'

    else:
        raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))
    
    return opt


def set_model(model_path):
    model = SupConResNet(name='resnet50') # for SupCon and SimCLR
    #model =  SupCEResNet(name='resnet50',num_classes=7) # for CE

    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model


def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = load_data(opt)

    # load models
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = set_model(opt.model_path)
    model.eval()

    emb_by_class =  {}
    for i in range(opt.n_cls):
        emb_by_class[i] = list()


    for images, labels in test_loader:
        images = images.float().cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            embeddings = model.encoder(images)

        # embeddings grouping for GS calculation
        emb_dict = group_embeddings(embeddings, labels)
        for class_name, embeddings_list in emb_dict.items():
            emb_by_class[class_name] += embeddings_list

    del train_loader, test_loader

    # calculate global separation by class
    print('Calculating global separation...')
    gs_values = calculate_global_separation(emb_by_class)
    for gs_class, gs_value in gs_values.items():
        print('class: {} \t global separation: {}'.format(gs_class, gs_value))

    # Calculate global separation by k-means clusters
    emb_array = [np.concatenate(emb).reshape(-1, 2048) for emb in emb_by_class.values()]
    emb_array = np.concatenate(emb_array)

    emb_array_normalized = normalize(emb_array, norm='l2')
    pca = PCA(n_components=128)
    emb_array_pca = pca.fit_transform(emb_array_normalized)

    num_cluster = [5,10,15,20,25,30]
    
    # for kmeans 
    
    clusters_A = apply_emb_clustering(emb_array_pca, K=num_cluster[0],algorithm='kmeans')
    clusters_B = apply_emb_clustering(emb_array_pca, K=num_cluster[1],algorithm='kmeans')
    clusters_C = apply_emb_clustering(emb_array_pca, K=num_cluster[2],algorithm='kmeans')
    clusters_D = apply_emb_clustering(emb_array_pca, K=num_cluster[3],algorithm='kmeans')
    clusters_E = apply_emb_clustering(emb_array_pca, K=num_cluster[4],algorithm='kmeans')
    clusters_F = apply_emb_clustering(emb_array_pca, K=num_cluster[5],algorithm='kmeans')

    '''
    # for GMM 
    clusters_A = apply_gmm_clustering(emb_array_pca, K=num_cluster[0])
    clusters_B = apply_gmm_clustering(emb_array_pca, K=num_cluster[1])
    clusters_C = apply_gmm_clustering(emb_array_pca, K=num_cluster[2])
    clusters_D = apply_gmm_clustering(emb_array_pca, K=num_cluster[3])
    clusters_E = apply_gmm_clustering(emb_array_pca, K=num_cluster[4])
    clusters_F = apply_gmm_clustering(emb_array_pca, K=num_cluster[5])
    '''

    gs_A = calculate_global_separation(clusters_A)
    gs_B = calculate_global_separation(clusters_B)
    gs_C = calculate_global_separation(clusters_C)
    gs_D = calculate_global_separation(clusters_D)
    gs_E = calculate_global_separation(clusters_E)
    gs_F = calculate_global_separation(clusters_F)

    cluster = ['GT'] * 7
    for i in range(len(num_cluster)):
        sequence = [num_cluster[i]] * num_cluster[i]
        cluster += sequence


    global_sep = list(gs_values.values()) + list(gs_A.values()) + list(gs_B.values()) + list(gs_C.values()) + list(gs_D.values()) + list(gs_E.values()) + list(gs_F.values()) 

    print('global_sep=', global_sep)
    print('cluster=', cluster)
    gs_results_df = pd.DataFrame({'GS': global_sep, 
                               'cluster': cluster})

    gs_results_df.to_csv('./results/{}/gs_results_dmf.csv'.format(opt.method))

    sns.boxplot(x='cluster', y='GS', data=gs_results_df)
    plt.savefig('./results/{}/gs_result_dmf.png'.format(opt.method))
    plt.show()

    # Calculate cluster purity by k-means clusters
    emb_by_class = {class_name: np.concatenate(emb).reshape(-1, 2048) for class_name, emb in emb_by_class.items()}
    emb_by_class_normalized = {class_name: normalize(emb) for class_name, emb in emb_by_class.items()}
    emb_by_class_pca = {class_name: pca.transform(emb) for class_name, emb in emb_by_class_normalized.items()}

    cp_A = calculate_cluster_purity(emb_by_class_pca, clusters_A)
    cp_B = calculate_cluster_purity(emb_by_class_pca, clusters_B)
    cp_C = calculate_cluster_purity(emb_by_class_pca, clusters_C)
    cp_D = calculate_cluster_purity(emb_by_class_pca, clusters_D)
    cp_E = calculate_cluster_purity(emb_by_class_pca, clusters_E)
    cp_F = calculate_cluster_purity(emb_by_class_pca, clusters_F)

    cluster_pur = [1.0]*7 + list(cp_A.values()) + list(cp_B.values()) + list(cp_C.values()) + list(cp_D.values()) + list(cp_E.values()) + list(cp_F.values())

    #print(cluster_pur)
    cp_results_df = pd.DataFrame({'CP': cluster_pur,
                               'cluster': cluster})
    
    cp_results_df.to_csv('./results/{}/cp_results_dmf.csv'.format(opt.method))

    sns.boxplot(x='cluster', y='CP', data=cp_results_df)
    plt.savefig('./results/{}/cp_result_dmf.png'.format(opt.method))
    plt.show()
    
    
if __name__ == '__main__':
    main()
