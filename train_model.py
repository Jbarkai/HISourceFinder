
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_generators.data_loader import SegmentationDataSet
import argparse
from os import listdir
import os
import shutil
from medzoo_imports import create_model, DiceLoss, Trainer
from datetime import datetime
from random import sample
import random
import gc
import numpy as np



def main(
    batch_size, shuffle, num_workers, dims, overlaps, root,
    random_seed, train_size, loaded, model, opt, lr, inChannels,
    classes, log_dir, dataset_name, terminal_show_freq, nEpochs,
    cuda, scale, subsample):
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
    print(loaded)
    if loaded:
        list_files = listdir(root+"generated/"+scale+"/_vol_128x128x64_"+str(int(train_size*100)))
        cubes = np.unique([i.split("_subcube")[0]+"_" for i in list_files])
    else:
        inputs = [root+scale+'Input/' + x for x in listdir(root+scale+'Input') if ".fits" in x]
        targets = [root+'Target/mask_' + x.split("/")[-1].split("_")[-1] for x in inputs]
        dataset_full = SegmentationDataSet(inputs=inputs,
                                            targets=targets,
                                            dims=dims,
                                            overlaps=overlaps,
                                            load=False,
                                            root=root,
                                            scale=scale)
        cubes = np.unique([i[0].split(dataset_full.sub_vol_path)[-1].split("_subcube")[0]+"_" for i in dataset_full.list])
    inputs_train, inputs_valid, targets_train, targets_valid, inputs_test, targets_test = [], [], [], [], [], []
    cubes = sample(list(cubes), subsample)
    for cube in cubes:
        if loaded:
            direct = root+"generated"+"/_vol_128x128x64_"+scale+"/"
            noisey = [direct+x for x in list_files if cube in x and "seg" not in x]
        else:
            noisey = [x[0] for x in dataset_full.list if cube in x[0]]
        
        num_test_val = int(len(noisey)*(1 - train_size)/2)
        num_train =int(len(noisey)*train_size)
        random.shuffle(noisey)
        inputs_tr = noisey[:num_train]
        targets_tr = [i.split(".npy")[0]+"seg.npy" for i in inputs_train]
        inputs_v = noisey[num_train:num_train+num_test_val]
        targets_v = [i.split(".npy")[0]+"seg.npy" for i in inputs_v]
        inputs_te = noisey[num_train+num_test_val:num_train+2*num_test_val]
        targets_te = [i.split(".npy")[0]+"seg.npy" for i in inputs_test]

        inputs_train.append(inputs_tr)
        inputs_valid.append(inputs_v)
        inputs_test.append(inputs_te)
        targets_train.append(targets_tr)
        targets_valid.append(targets_v)
        targets_test.append(targets_te)
    inputs_train = [item for sublist in inputs_train for item in sublist]
    inputs_valid = [item for sublist in inputs_valid for item in sublist]
    inputs_test = [item for sublist in inputs_test for item in sublist]
    targets_train = [item for sublist in targets_train for item in sublist]
    targets_valid = [item for sublist in targets_valid for item in sublist]
    targets_test = [item for sublist in targets_test for item in sublist]
    # dataset training
    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        dims=dims,
                                        overlaps=overlaps,
                                        load=True,
                                        root=root,
                                        mode="train",
                                        scale=scale)

    # dataset validation
    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        dims=dims,
                                        overlaps=overlaps,
                                        load=True,
                                        root=root,
                                        mode="val",
                                        scale=scale)

    # dataset validation
    dataset_test = SegmentationDataSet(inputs=inputs_test,
                                        targets=targets_test,
                                        dims=dims,
                                        overlaps=overlaps,
                                        load=True,
                                        root=root,
                                        mode="test",
                                        scale=scale)
    del inputs_train
    del inputs_valid
    del targets_train
    del targets_valid
    gc.collect()
    now = datetime.now() # current date and time
    date_str = now.strftime("%d%m%Y_%H%M%S")
    save = ('./saved_models/' + model + '_checkpoints/' + model + '_', dataset_name + "_" + date_str)[0]
    # dataloader training
    params = {'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers}
    dataloader_training = DataLoader(dataset=dataset_train, **params)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid, **params)
    del dataset_train
    del dataset_valid
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
    return trainer


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
    args = parser.parse_args()

    main(
        args.batch_size, args.shuffle, args.num_workers, args.dims,
        args.overlaps, args.root, args.random_seed, args.train_size, args.loaded,
        args.model, args.opt, args.lr, args.inChannels, args.classes,
        args.log_dir, args.dataset_name, args.terminal_show_freq,
        args.nEpochs, args.cuda, args.scale, args.subsample)
