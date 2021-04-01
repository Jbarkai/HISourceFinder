import torch
from skimage.io import imread
from torch.utils import data
from astropy.io import fits
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib
from os import listdir


class HIDataSet(data.Dataset):
    def __init__(self,
                 inputs: list, # list of input paths
                 targets: list, # list of mask paths
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        cube_hdulist = fits.open(input_ID)
        x = cube_hdulist[0].data
        cube_hdulist.close()
        maskcube_hdulist = fits.open(target_ID)
        y = maskcube_hdulist[0].data
        maskcube_hdulist.close()

        # Preprocessing
        # if self.transform is not None:
        #     x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x.astype(np.float32)).type(self.inputs_dtype), torch.from_numpy(y.astype(np.int64)).type(self.targets_dtype)

        return x, y
        # t1_path, ir_path, flair_path, seg_path = self.list[index]
        # return np.load(t1_path), np.load(ir_path), np.load(flair_path), np.load(seg_path)

def generate_datasets(args, root='../data/training/'):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    
    # input and target files
    inputs = listdir(root + '/Input')
    targets = listdir(root + '/Target')

    # random seed
    random_seed = args.random_seed # 42
    # split dataset into training set and validation set
    train_size = args.split# 0.8  # 80:20 split
    transforms = args.transforms
    samples_train = args.samples_train
    samples_val = args.samples_val

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=random_seed,
        train_size=train_size,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=random_seed,
        train_size=train_size,
        shuffle=True)
    # dataset training
    dataset_train = HIDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)
    # dataset validation
    dataset_valid = HIDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=transforms)

    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=2,
                                    shuffle=True)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=2,
                                    shuffle=True)
    # training_generator = DataLoader(train_loader, **params)
    # val_generator = DataLoader(val_loader, **params)
    x, y = next(iter(dataloader_training))

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return dataloader_training, dataloader_validation, val_loader.full_volume, val_loader.affine