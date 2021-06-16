
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_generators.data_loader import SegmentationDataSet
import argparse
from os import listdir
import os
import torch
import shutil
from medzoo_imports import create_model, DiceLoss, Trainer
from datetime import datetime
from random import sample
import random
import copy
from scipy import ndimage as ndi
import gc
import pickle
import numpy as np


def main(
    batch_size, shuffle, num_workers, dims, overlaps, root,
    random_seed, train_size, model, opt, lr, inChannels,
    classes, log_dir, dataset_name, terminal_show_freq, nEpochs,
    cuda, scale, subsample, k_folds, pretrained, load_data_loc):
    """Create training and validation datasets

    Args:
        batch_size (int): Batch size
        shuffle (bool): Whether or not to shuffle the train/val split
        num_workers (int): The number of workers to use
        dims (list): The dimensions of the subcubes
        overlaps (list): The dimensions of the overlap  of subcubes
        root (str): The root directory of the data
        random_seed (int): Random Seed
        train_size (float): Ratio of training to validation split
        model (str): The 3D segmentation model to use
        opt (str): The type of optimizer
        lr (float): The learning rate
        inChannels (int): The desired modalities/channels that you want to use
        classes (int): The number of classes
        log_dir (str): The directory to output the logs
        dataset_name (str): The name of the dataset
        terminal_show_freq (int): How often it shows the output
        nEpochs (int): The number of epochs
        scale (str): Loud or soft - S-N ratio
        subsample (int): Size of subset
        k_folds (int): The number of folds for cross validation
        pretrained (str): The location of the pretrained model

    Returns:
        The training and validation data loaders
    """
    now = datetime.now() # current date and time
    date_str = now.strftime("%d%m%Y_%H%M%S")
    # input and target files
    model_name = model
    model, optimizer = create_model(args)
    if pretrained:
        # model.restore_checkpoint(pretrained)
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        save = pretrained.split("/")[0] + "/"
    else:
        start_epoch = 0
        save = "./saved_models_%s_%s_%s/"%(date_str, scale, subsample)
        if not os.path.exists(save):
            os.mkdir(save)
        with open(save+"params.txt", "wb") as fp:
            pickle.dump([{
                "scale": scale,
                "subsample": subsample,
                "opt":opt,
                "lr": lr,
                "batch_size": batch_size
            }], fp)
    inputs = [root+scale+'Input/' + x for x in listdir(root+scale+'Input') if ".fits" in x]
    targets = [root+'Target/mask_' + x.split("/")[-1].split("_")[-1] for x in inputs]
    dataset_full = SegmentationDataSet(inputs=inputs,
                                        targets=targets,
                                        dims=dims,
                                        overlaps=overlaps,
                                        root=root,
                                        mode="full",
                                        save_name=save)
    # dataset validation
    dataset_test = copy.deepcopy(dataset_full)
    dataset_test.list = [i for i in dataset_full.list if "_1245mos" in i[0][0]]
    if subsample < 7:
        test_cubes = [i.split("/")[-1] for i in inputs if "_1245mos" in i]
        test_cubes = sample(test_cubes, subsample)
        dataset_test.list = [j for j in dataset_test.list if j[0][0].split("/")[-1] in test_cubes]
    print(len(dataset_test.list))
    params = {'batch_size': 1,
            'shuffle': shuffle,
            'num_workers': num_workers}
    # dataloader test
    dataloader_test = DataLoader(dataset=dataset_test, **params)
    with open(save+'test--hisource-list-slidingwindowindices.txt', "wb") as fp:
        pickle.dump(dataset_test.list, fp)
    # dataset_test.list = dataset_test.list[:10]
    print(len(dataset_test.list))
    cubes = [i.split("/")[-1] for i in inputs if "_1353mos" in i]
    if subsample < 7:
        cubes = sample(cubes, subsample)
    # For fold results
    results = {}
    print('--------------------------------')
    for k in range(k_folds):
        print('FOLD %s'%k)
        print('--------------------------------')
        args.save = (save + 'fold_' + str(k) + '_checkpoints/' + model_name + '_', dataset_name + "_" + date_str)[0]
        if load_data_loc == "":
            train_list, val_list = [], []
            for cube in cubes:
                file_list = [i for i in dataset_full.list if cube in i[0][0]]
                random.shuffle(file_list)
                print(len(file_list))
                num_val = int(len(file_list)*(1 - train_size))
                num_train =int(len(file_list)*train_size)
                print(num_train, num_val)
                train_list +=(file_list[:num_train])
                val_list +=(file_list[num_train:num_train+num_val])
            if not os.path.exists(save + 'fold_' + str(k) + '_checkpoints'):
                os.mkdir(save + 'fold_' + str(k) + '_checkpoints')
            with open(save + 'fold_' + str(k) + '_checkpoints/train_windows.txt', "wb") as fp:
                pickle.dump(train_list, fp)
            with open(save + 'fold_' + str(k) + '_checkpoints/val_windows.txt', "wb") as fp:
                pickle.dump(val_list, fp)
        else:
            with open(load_data_loc + '/train_windows.txt', "rb") as fp:
                train_list = pickle.load(fp)
            with open(load_data_loc + '/val_windows.txt', "rb") as fp:
                val_list = pickle.load(fp)
        # dataset training
        dataset_train = copy.deepcopy(dataset_full)
        dataset_train.list = train_list
        # dataset validation
        dataset_valid = copy.deepcopy(dataset_full)
        dataset_valid.list = val_list
        # dataset_train.list = dataset_train.list[:10]
        # dataset_valid.list = dataset_valid.list[:10]
        # dataloader training
        params = {'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers}
        dataloader_training = DataLoader(dataset=dataset_train, **params)
        # dataloader validation
        dataloader_validation = DataLoader(dataset=dataset_valid, **params)
        print(dataloader_training.__len__(), dataloader_validation.__len__(), dataloader_test.__len__())
        criterion = DiceLoss(classes=args.classes)
        trainer = Trainer(args, model, criterion, optimizer, train_data_loader=dataloader_training,
                                valid_data_loader=dataloader_validation, lr_scheduler=None, start_epoch=start_epoch)
        print("START TRAINING...")
        trainer.training()

        # Evaluation for this fold
    #     model.eval()
    #     intersections = 0
    #     all_or = 0
    #     with torch.no_grad():
    #         for batch_idx, input_tuple in enumerate(dataloader_test):
    #             input_tensor, target = input_tuple
    #             out_cube = model.inference(input_tensor)
    #             out_np = out_cube.squeeze()[1].numpy()
    #             target_np = target.squeeze().numpy()
    #             # Turn probabilities to mask
    #             smoothed_gal = ndi.gaussian_filter(out_np, sigma=2)
    #             # Relabel each object seperately
    #             t = np.nanmean(smoothed_gal) + np.nanstd(smoothed_gal)
    #             new_mask = (smoothed_gal > t)
    #             gt = (target_np).flatten().tolist()
    #             pred = (new_mask).flatten().tolist()
    #             intersections += np.nansum(np.logical_and(gt, pred).astype(int))
    #             all_or += np.nansum(gt) + np.nansum(pred)
    #         dice_losses = 2*intersections/all_or
    #         # Print accuracy
    #         print('Total dice loss for fold ', k , ":", (100.0*dice_losses), "%")
    #         print('--------------------------------')
    #         results[k] = 100.0*dice_losses
    
    # with open(save + "vnet_dice.txt", "wb") as fp:
    #     pickle.dump(results, fp)
    # # Print fold results
    # print('K-FOLD CROSS VALIDATION RESULTS FOR %s FOLDS'%k_folds)
    # print('--------------------------------')
    # for key, value in results.items():
    #     print('Fold ', key, ":", value, " %")
    # av = np.mean([i for i in results.values()])
    # std = np.std([i for i in results.values()])
    # print('Average: ', av, "%")
    # print('Standard Deviation: ', std, "%")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--batch_size', type=int, nargs='?', const='default', default=4,
        help='Batch size')
    parser.add_argument(
        '--shuffle', type=bool, nargs='?', const='default', default=True,
        help='Whether or not to shuffle the train/val split')
    parser.add_argument(
        '--num_workers', type=int, nargs='?', const='default', default=2,
        help='The number of workers to use')
    parser.add_argument(
        '--dims', type=list, nargs='?', const='default', default=[128, 128, 64],
        help='The dimensions of the subcubes')
    parser.add_argument(
        '--overlaps', type=list, nargs='?', const='default', default=[15, 20, 20],
        help='The dimensions of the overlap of subcubes')
    parser.add_argument(
        '--root', type=str, nargs='?', const='default', default='./data/training/',
        help='The root directory of the data')
    parser.add_argument(
        '--random_seed', type=int, nargs='?', const='default', default=42,
        help='Random Seed')
    parser.add_argument(
        '--train_size', type=float, nargs='?', const='default', default=0.8,
        help='Ratio of training to validation split')
    parser.add_argument(
        '--model', type=str, nargs='?', const='default', default='VNET',
        help='The 3D segmentation model to use')
    parser.add_argument(
        '--opt', type=str, nargs='?', const='default', default='adam',
        help='The type of optimizer')
    parser.add_argument(
        '--lr', type=float, nargs='?', const='default', default=1e-3,
        help='The learning rate')
    parser.add_argument(
        '--inChannels', type=int, nargs='?', const='default', default=1,
        help='The desired modalities/channels that you want to use')
    parser.add_argument(
        '--classes', type=int, nargs='?', const='default', default=2,
        help='The number of classes')
    parser.add_argument(
        '--log_dir', type=str, nargs='?', const='default', default="./runs/",
        help='The directory to output the logs')
    parser.add_argument(
        '--dataset_name', type=str, nargs='?', const='default', default='hi_source',
        help='The name of the dataset')
    parser.add_argument(
        '--terminal_show_freq', type=int, nargs='?', const='default', default=500,
        help='Show when to print progress')
    parser.add_argument(
        '--nEpochs', type=int, nargs='?', const='default', default=10,
        help='The number of epochs')
    parser.add_argument(
        '--scale', type=str, nargs='?', const='default', default="",
        help='The scale of inserted galaxies to noise')
    parser.add_argument(
        '--subsample', type=int, nargs='?', const='default', default=10,
        help='The size of subset to train on')
    parser.add_argument(
        '--cuda', type=bool, nargs='?', const='default', default=False,
        help='Memory allocation')
    parser.add_argument(
        '--k_folds', type=int, nargs='?', const='default', default=5,
        help='Number of folds for k folds cross-validations')
    parser.add_argument(
        '--pretrained', type=str, nargs='?', const='default', default=None,
        help='The location of the pretrained model')
    parser.add_argument(
        '--load_data_loc', type=str, nargs='?', const='default', default="",
        help='The location of the data windows')
    args = parser.parse_args()

    main(
        args.batch_size, args.shuffle, args.num_workers, args.dims,
        args.overlaps, args.root, args.random_seed, args.train_size,
        args.model, args.opt, args.lr, args.inChannels, args.classes,
        args.log_dir, args.dataset_name, args.terminal_show_freq,
        args.nEpochs, args.cuda, args.scale, args.subsample, args.k_folds,
        args.pretrained, args.load_data_loc)
