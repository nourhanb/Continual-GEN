from __future__ import print_function
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from hdbscan import HDBSCAN
import math
import numpy as np
import pandas as pd
import torch
import copy
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
 

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def group_embeddings(batch_embeddings, labels):
    labels = labels.cpu().numpy()
    batch_embeddings.cpu().detach().numpy()

    # Grouping embeddings by their class (cluster)
    embeddings_by_class = {}

    for i in range(len(labels)):
        class_label = labels[i]
        embedding = batch_embeddings[i].cpu().detach().numpy()
        if class_label in embeddings_by_class:
            embeddings_by_class[class_label].append(embedding)
        else:
            embeddings_by_class[class_label] = [embedding]

    return embeddings_by_class


def calculate_global_separation(embeddings_by_class, X=0.06):
    """Compute global separation per class.
    Args:
        embeddings_by_class: dictionary with class numbers as keys and lists of embeddings as values.
        X: Percentage of samples used in the calculations. Between 0 and 1.
    Returns:
        A dictionary with class numbers as keys and their GS as values.
    """
    # Nested lists of classes and their corresponding embeddings
    class_list = list(embeddings_by_class.keys())
    embed_list = list(embeddings_by_class.values())

    # Obtaing the nearest class to each one
    mean_embeddings_by_class = [np.mean(v, axis=0) for v in embed_list]
    distances = euclidean_distances(mean_embeddings_by_class)

    for i in range(distances.shape[0]):
        distances[i,i] = 10.0

    nearest_classes = np.argmin(distances, axis=1)

    # Finally, GS is calculated for each class (cluster) present in the batch
    gs_per_class = {}
    
    for i in range(len(class_list)):
        intra_distances = euclidean_distances(embed_list[i])
        intra_distances = np.sort(np.unique(intra_distances)[1:])
        intra_idx = round(len(intra_distances) * X) #Taking the smallest x% samples
        intra_distances = np.mean(intra_distances[:intra_idx])
    
        nearest_emb_idx = nearest_classes[i]
        inter_distances = euclidean_distances(embed_list[i], embed_list[nearest_emb_idx])
        inter_distances = np.sort(inter_distances)
        inter_idx = round(len(inter_distances) * X)
        inter_distances = np.mean(inter_distances[:inter_idx])

        gs = (inter_distances - intra_distances) / max(inter_distances, intra_distances)
        gs_per_class[class_list[i]] = round(gs, 3)

    return gs_per_class


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from hdbscan import HDBSCAN

def apply_emb_clustering(emb_array, K, algorithm):
    """Perform clustering over a group of embeddings.
    Args:
        emb_array: array of embeddings conforming each row and 2048 columns (emb dimension).
        K: number of desired clusters.
        algorithm: clustering algorithm to use (e.g., 'kmeans', 'agglomerative', 'dbscan', 'spectral', 'meanshift', 'hdbscan')
    Returns:
        A dictionary with cluster numbers as keys and embeddings lists as values.
    """
    if algorithm == 'kmeans':
        clustering = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(emb_array)
    elif algorithm == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=K).fit(emb_array)
    elif algorithm == 'dbscan':
        clustering = DBSCAN().fit(emb_array)
    elif algorithm == 'spectral':
        clustering = SpectralClustering(n_clusters=K).fit(emb_array)
    elif algorithm == 'meanshift':
        clustering = MeanShift().fit(emb_array)
    elif algorithm == 'hdbscan':
        clustering = HDBSCAN(min_cluster_size=K).fit(emb_array)
    else:
        raise ValueError("Invalid clustering algorithm.")

    df = pd.DataFrame({'Embedding': emb_array.tolist(),
                       'Cluster': clustering.labels_})

    embeddings_by_cluster = {}
    for cluster in set(clustering.labels_):
        embeddings_by_cluster[cluster] = list(df.loc[df['Cluster'] == cluster]['Embedding'])

    return embeddings_by_cluster

def apply_gmm_clustering(emb_array, K,  covariance ='full', method='random_from_data'):
    """Perform Gaussian Mixture over a group of embeddings.
    Args:
        emb_array: array of embeddings conforming each row and 2048 columns (emb dimension).
        K: number of desired clusters.
        method: The method used to initialize the weights, the means and the precisions. {'kmeans', 'k-means++', 'random', 'random_from_data'}
    Returns:
        A dictionary with cluster numbers as keys and embeddings lists as values.
    """
    gm = GaussianMixture(n_components=K, random_state=0, covariance_type = covariance, init_params=method).fit_predict(emb_array)

    df = pd.DataFrame({'Embedding': emb_array.tolist(),
              'Cluster': gm})
    
    embeddings_by_cluster = {}
    for cluster in gm:
        embeddings_by_cluster[cluster] = list(df.loc[df['Cluster'] == cluster]['Embedding'])

    return embeddings_by_cluster

def calculate_cluster_purity(emb_by_class, emb_by_cluster):
    """Compute cluster purity.
    Args:
        emb_by_class: dictionary with class numbers as keys and lists of embeddings 
                    as values (returned from group_embeddings function).
        emb_by_cluster: dictionary with cluster numbers as keys and lists of embeddings 
                    as values (returned from apply_emb_clustering function).
    Returns:
        A dictionary with class numbers as keys and their CP as values.
    """
    cp_by_cluster = {}

    for cluster, cluster_emb in emb_by_cluster.items():
        cluster_emb = np.mean(cluster_emb, axis=1).round(5)
        
        class_count = list()
        for class_emb in emb_by_class.values():
            class_emb = np.mean(class_emb, axis=1).round(5)
            count = len([emb for emb in cluster_emb if emb in class_emb])
            class_count.append(count)

        cp = max(class_count) / len(cluster_emb)
        cp_by_cluster[cluster] = round(cp, 3)

    return cp_by_cluster

