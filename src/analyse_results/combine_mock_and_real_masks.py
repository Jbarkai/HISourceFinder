from astropy.io import fits
from os import listdir
from astropy.io import fits
import numpy as np

for mask in [i for i in listdir("./Target/") if (".fits" in i) & ("1353" in i)]:
    print(mask)
    mock_mask = fits.getdata("Target/"+mask)
    real_mask = fits.getdata("TargetReal/"+mask)
    new_mask = mock_mask + real_mask
    new_mask[new_mask > 0] = 1
    fits.writeto("TargetBoth/%s"%mask, new_mask, overwrite=True)
