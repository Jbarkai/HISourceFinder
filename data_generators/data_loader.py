import torch
from skimage.io import imread
from torch.utils import data
from astropy.io import fits


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
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x.astype(np.float32)).type(self.inputs_dtype), torch.from_numpy(y.astype(np.int64)).type(self.targets_dtype)

        return x, y