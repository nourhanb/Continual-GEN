from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import pandas as pd

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from util import AverageMeter, TwoCropTransform
from util import adjust_learning_rate, warmup_learning_rate, accuracy, calculate_global_separation, group_embeddings
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

from ham import create_dataframes, create_transformations, HAM10000, compute_img_mean_std

import warnings
# Filter out DeprecationWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ensure results are reproducible
np.random.seed(117)
torch.manual_seed(117)
torch.cuda.manual_seed(117)


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='15',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '/ubc/ece/home/ra/grads/nourhanb/Documents/skins/DMF'
    opt.model_path = './save/{}/DMF_models'.format(opt.method)
    opt.tb_path = './save/{}/DMF_tensorboard'.format(opt.method)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, 'DMF', opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_cls = 7
    opt.input_size = 100

    return opt


def load_data(opt):
    # Images means and standard deviations of their three RGB channels
    norm_mean, norm_std = compute_img_mean_std(opt.data_folder)

    df_train, df_test = create_dataframes(opt.data_folder)

    # define the transformation of the train images.
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=50, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]
        )

    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((opt.input_size,opt.input_size)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    # Define the training set using the table df_train and using the defined transitions (train_transform)
    training_set = HAM10000(df_train, transform=TwoCropTransform(train_transform))
    train_loader = DataLoader(training_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # Same for the test set:
    test_set = HAM10000(df_test, transform=val_transform)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    return train_loader, test_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def gs_validation(val_loader, model, opt):
    """one epoch training"""
    model.eval()

    embeddings_by_class =  {}
    for i in range(opt.n_cls):
        embeddings_by_class[i] = list()

    for idx, (images, labels) in enumerate(val_loader):
        images = images.float().cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            embeddings = model.encoder(images)

        # embeddings grouping for GS calculation
        embeddings_dict = group_embeddings(embeddings, labels)

        for class_name, embeddings_list in embeddings_dict.items():
            embeddings_by_class[class_name] += embeddings_list

    return embeddings_by_class

def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = load_data(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    history = pd.DataFrame(columns=['epoch', 'train_loss', '0_GS','1_GS','2_GS','3_GS','4_GS','5_GS','6_GS'])

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # calculate global separation by class
        embeddings_by_class = gs_validation(test_loader, model, opt)
        gs_values = calculate_global_separation(embeddings_by_class)
        for gs_class, gs_value in gs_values.items():
            gs_value = np.array(gs_value)
            gs_value = gs_value[~np.isnan(gs_value)]
            gs_values[gs_class] = np.mean(gs_value)
            print('class: {} \t global separation: {}'.format(gs_class, gs_values[gs_class]))

        # save history
        record = [epoch, loss] + list(gs_values.values())
        record = pd.DataFrame([record], columns=history.columns)
        history = pd.concat([history, record], ignore_index=True)
        save_hist = os.path.join(opt.save_folder, 'history.csv')
        history.to_csv(save_hist)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
