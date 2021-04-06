import torch
from skimage.io import imread
from torch.utils.data import Dataset
from astropy.io import fits
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib
from os import listdir


class SegmentationDataSet(Dataset):
    def __init__(self,
                 inputs: list, # list of input paths
                 targets: list, # list of mask paths
                 dims=[10, 500, 500],
                 overlaps=[8, 400, 400],
                 load=False,
                 mode='train',
                 root='../HISourceFinder/data/training/'
                 ):
        self.list = []
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.dims = dims
        self.overlaps = overlaps
        self.save_name = root + 'hisource-list-slidingwindow.txt'
        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            return

        subvol = '_vol_' + str(dims[0]) + 'x' + str(dims[1]) + 'x' + str(dims[2])
        self.sub_vol_path = root + '/generated/' + subvol + '/'
        utils.make_dirs(self.sub_vol_path)
        ################ SLIDING WINDOW ######################
        for index in range(len(self.inputs)):
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load and slide over input
            cube_hdulist = fits.open(input_ID)
            x = cube_hdulist[0].data
            cube_hdulist.close()
            tensor_images = sliding_window(x, dims, overlaps)

            # Load and slide over target
            maskcube_hdulist = fits.open(target_ID)
            y = maskcube_hdulist[0].data
            maskcube_hdulist.close()
            tensor_segs = sliding_window(y, dims, overlaps)
            filename = self.sub_vol_path + 'cube_' + str(index) +"_subcube_"
            list_saved_paths = [(filename + str(j) + '.npy', filename + str(j) + 'seg.npy') for j in range(len(tensor_images))]
            ############### SAVE SUBCUBES ##########################
            for j in range(len(tensor_images)):
                np.save(list_saved_paths[j][0], tensor_images[j])
                np.save(list_saved_paths[j][1], tensor_segs[j])
            self.list += list_saved_paths
        # Save list of subcubes
        with open(self.save_name, "wb") as fp:
            pickle.dump(self.list, fp)

    def __len__(self):
        return len(self.list)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_path, seg_path = self.list[index]
        x, y = np.load(input_path), np.load(seg_path)
        # Typecasting
        input_x, target_y = torch.from_numpy(
            x.astype(np.float32)).type(self.inputs_dtype),
        torch.from_numpy(y.astype(np.int64)).type(self.targets_dtype)
        return input_x, target_y

def sliding_window(arr, dims, overlaps):
    kernel=(1, dims[0], dims[1], dims[2], 1)
    stride=(1, overlaps[0], overlaps[1], overlaps[2], 1) 
    _,sx,sy,sz,_ = kernel   
    in_patches=tf.extract_volume_patches(
        arr[None, ..., None],kernel,stride,'SAME',
    )
    _,x,y,z,n = in_patches.shape
    in_patches = tf.reshape(in_patches,[x*y*z,sx,sy,sz])
    return in_patches

# input and target files
inputs = ['../data/training/Input/' + x for x in listdir('../data/training/Input') if ".fits" in x]
targets = ['../data/training/Target/' + x for x in listdir('../data/training/Target') if ".fits" in x]
# random seed
random_seed = 42
# split dataset into training set and validation set
train_size = 0.8  # 80:20 split

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
dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets=targets_train,
                                    dims=[10, 500, 500],
                                    overlaps=[8, 400, 400],
                                    load=False,
                                    mode='train',
                                    root='../HISourceFinder/data/training/')

# dataset validation
dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                    targets=targets_valid,
                                    dims=[10, 500, 500],
                                    overlaps=[8, 400, 400],
                                    load=False,
                                    mode='train',
                                    root='../HISourceFinder/data/training/')

# dataloader training
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 2}
dataloader_training = DataLoader(dataset=dataset_train, **params)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid, **params)

###### TRAIN MODEL ############
# class argsclass:
#     def __init__(self, model, opt, lr, inChannels, classes):
#         self.model = model
#         self.opt=opt
#         self.lr=lr
#         self.inChannels=inChannels
#         self.classes=classes
#         self.log_dir = "./runs/"
#         self.dataset_name = 'iseg2017'
#         self.save = save
#         self.terminal_show_freq = 50
#         self.nEpochs = 10
# args = argsclass('VNET', opt, lr, inChannels, classes)
# model, optimizer = medzoo.create_model(args)
# criterion = DiceLoss(classes=args.classes)
# utils.make_dirs(save)
# trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=dataloader_training,
#                         valid_data_loader=dataloader_validation, lr_scheduler=None)