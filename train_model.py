
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
from scipy import ndimage as ndi
import gc
import numpy as np


def main(
    batch_size, shuffle, num_workers, dims, overlaps, root,
    random_seed, train_size, loaded, model, opt, lr, inChannels,
    classes, log_dir, dataset_name, terminal_show_freq, nEpochs,
    cuda, scale, subsample, k_folds):
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
        terminal_show_freq (int): 
        nEpochs (int): The number of epochs
        scale (str): Loud or soft - S-N ratio
        subsample (int): Size of subset
        loaded (bool): Whether to load pre-generated data

    Returns:
        The training and validation data loaders
    """
    # input and target files
    model_name = model
    inputs = [root+scale+'Input/' + x for x in listdir(root+scale+'Input') if ".fits" in x]
    targets = [root+'Target/mask_' + x.split("/")[-1].split("_")[-1] for x in inputs]
    dataset_full = SegmentationDataSet(inputs=inputs,
                                        targets=targets,
                                        dims=dims,
                                        overlaps=overlaps,
                                        load=loaded,
                                        root=root)
        
    cubes = [i.split("/")[-1] for i in inputs]
    cubes = sample(cubes, subsample)
    # For fold results
    results = {}
    print('--------------------------------')
    for k in range(k_folds):
        print('FOLD %s'%k)
        print('--------------------------------')
        train_list, val_list, test_list = [], [], []
        for cube in cubes:
            file_list = [i for i in dataset_full.list if cube in i[0][0]]
            num_test_val = int(len(file_list)*(1 - train_size)/2)
            num_train =int(len(file_list)*train_size)
            random.shuffle(file_list)
            train_list +=(file_list[:num_train])
            val_list +=(file_list[num_train:num_train+num_test_val])
            test_list +=(file_list[num_train+num_test_val:num_train+2*num_test_val])
        # dataset training
        dataset_train = SegmentationDataSet(inputs=inputs_train,
                                            targets=targets_train,
                                            dims=dims,
                                            overlaps=overlaps,
                                            load=True,
                                            root=root,
                                            list=train_list)

        # dataset validation
        dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                            targets=targets_valid,
                                            dims=dims,
                                            overlaps=overlaps,
                                            load=True,
                                            root=root,
                                            list=val_list)

        # dataset validation
        dataset_test = SegmentationDataSet(inputs=inputs_test,
                                            targets=targets_test,
                                            dims=dims,
                                            overlaps=overlaps,
                                            load=True,
                                            root=root,
                                            list=test_list)
        now = datetime.now() # current date and time
        date_str = now.strftime("%d%m%Y_%H%M%S")
        save = ('./saved_models/fold_' + str(k) + '_checkpoints/' + model_name + '_', dataset_name + "_" + date_str)[0]
        # dataloader training
        params = {'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers}
        dataloader_training = DataLoader(dataset=dataset_train, **params)
        # dataloader validation
        dataloader_validation = DataLoader(dataset=dataset_valid, **params)
        # dataloader test
        dataloader_test = DataLoader(dataset=dataset_test, **params)
        del dataset_train
        del dataset_valid
        del dataset_test
        gc.collect()
        model, optimizer = create_model(args)
        criterion = DiceLoss(classes=args.classes)
        if os.path.exists(save):
            shutil.rmtree(save)
            os.mkdir(save)
        else:
            os.makedirs(save)
        args.save = save
        trainer = Trainer(args, model, criterion, optimizer, train_data_loader=dataloader_training,
                                valid_data_loader=dataloader_validation, lr_scheduler=None)
        del dataloader_training
        del dataloader_validation
        gc.collect()
        print("START TRAINING...")
        trainer.training()

        # Evaluationfor this fold
        dice_losses, total = 0, 0
        model.eval()
        with torch.no_grad():
            for batch_idx, input_tuple in enumerate(dataloader_test):
                input_tensor, target = input_tuple
                out_cube = model.inference(input_tensor)# Grab in numpy array
                out_np = out_cube.squeeze()[0].numpy()
                target_np = target.squeeze()[0].numpy()
                # Turn probabilities to mask
                smoothed_gal = ndi.gaussian_filter(out_np, sigma=2)
                # Relabel each object seperately
                t = np.abs(np.mean(smoothed_gal))- np.std(smoothed_gal)
                new_mask = (smoothed_gal > t)
                intersection = np.nansum(np.logical_and(target_np, new_mask).astype(int))
                if np.nansum(target_np) == np.nansum(new_mask) == 0:
                    dice = 1
                else:
                    union = np.nansum(target_np) + np.nansum(new_mask)
                    dice = (2*intersection)/(union)
                total += batch_idx
                dice_losses += dice

            # Print accuracy
            print('Average dice loss for fold ', k , ":", (100.0*dice_losses/total), "%")
            print('--------------------------------')
            results[k] = 100.0 * (dice_losses / total)
    # Print fold results
    print('K-FOLD CROSS VALIDATION RESULTS FOR %s FOLDS'%k_folds)
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print('Fold ', key, ":", value, " %")
        sum += value
    av = sum/len(results.items())
    print('Average: ', av, "%")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--loaded', type=bool, nargs='?', const='default', default=False,
        help='Whether to load pre-generated data')
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
        '--train_size', type=float, nargs='?', const='default', default=0.6,
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
    args = parser.parse_args()

    main(
        args.batch_size, args.shuffle, args.num_workers, args.dims,
        args.overlaps, args.root, args.random_seed, args.train_size, args.loaded,
        args.model, args.opt, args.lr, args.inChannels, args.classes,
        args.log_dir, args.dataset_name, args.terminal_show_freq,
        args.nEpochs, args.cuda, args.scale, args.subsample, args.k_folds)
