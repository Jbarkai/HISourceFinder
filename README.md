# 3D HI Source Finder
<p align="center">
  <img src="https://github.com/Jbarkai/HISourceFinder/blob/master/cover_pic.png" width="500" />
</p>

A comparison of object segmentation techniques for optimal source finding in HI 3D data cubes.
Corresponding Author: Jordan A. Barkai, University of Groningen

Contributors (in alphabetical order): Jordan A. Barkai, Estefanía Talavera Martínez, Marc A.W. Verheijen, Michael H. F. Wilkinson
## Project Description
Astronomical surveys map the skies without a specific target, resulting in images containing many astronomical objects. As the technology used to create these surveys improves with projects like the SKA, an unprecedented amount of data will become available. Besides the time that it would require, manually detecting sources in data-sets of this volume would be unfeasible due tothe low intensities of many sources, increasing their proximity to the level of noise. Hence the need for fast and accurate techniques to detect and locate sources in astronomical survey data.

The problem at hand can be seen as an overlap between fields, where existing advances in computer vision could be used to solve the struggle of source finding in astronomy. In computer vision object recognition encompasses a collection of tasks, including object segmentation, which we will refer to as source finding. Object segmentation can be defined as drawing a mask around identified objects in an image and assigning them class labels. This is done by highlighting pixels (or voxels in the case of 3D data cubes) belonging to different objects such as astronomical sources and noise or background.

The challenge lies in the lack of clarity in the boundaries of sources, with many having intensities very close to the noise, especially in the case of radio data. Additionally, as the sensitivity and depth in astronomical surveys will increase, so will the number of overlapping sources as fainter and more extended sources are observed. This concept is known as blending. Having this foresight, many astronomers have explored source finding and deblending solutions using simple statistical techniques. However, these approaches are very sensitive to the input parameters and have been found to struggle with differentiating between faint sources and noise, resulting in a trade-off between accepting false sources and excluding true sources. While object segmentationin 2D images is considered a “solved” problem these days, the same task in 3D data cubes is still very new and unexplored in both astronomy and computer vision. 

In this project we will explore the various existing methods, including the traditional statistical approaches as well as machine learning techniques in attempt to create a pipeline to optimally mask and label the sources in 3D neutral hydrogen (HI) data cubes.


## Setup
Clone repository.
```bash
git clone https://github.com/Jbarkai/HISourceFinder.git
cd HISourceFinder
```
Create and activate a Python 3.7 environment.
```bash
conda create -n hisources python=3.7
source activate hisources
```
Install the required packages.
```bash
pip --cache-dir /tmp/pipcache install -r requirements.txt
```
## Data Directory Structure
The data directories need to be created as follows, but only the fits files of the synthetic galaxies under `mock_gals` and the noisy mosaic fits files under `mosaics` need to be there to start, the rest will be created by the scripts below.
```
data 
│
└───mock_gals
│   │   g1_model474.fits
│   │   g1_model475.fits
│   │   ...
└───mosaics
│   │   1245mosB.derip.fits
│   │   1245mosC.derip.fits
│   │   ...
└───mto_output
│   │   mtocubeout_loud_1245mosB.fits
│   │   mtocubeout_loud_1245mosC.fits
│   │   ...
└───vnet_output
│   │   vnet_cubeout_loud_1245mosB.fits
│   │   vnet_cubeout_loud_1245mosC.fits
│   │   ...
└───sofia_ouput
│   │   sofia_loud_1245mosC_mask.fits
│   │   sofia_loud_1245mosC_rel.eps
│   │   ...
└───training
│   └───NoiseFree
│       │   noisefree_1245mosB.fits
│       │   noisefree_1245mosC.fits
│       │   ...
│   └───Target
│       │   mask_1245mosB.fits
│       │   mask_1245mosC.fits
│       │   ...
│   └───Input
│       │   1245mosB.fits
│       │   1245mosC.fits
│       │   ...
```
## Usage
### Create Mock Cubes (all scripts in data_generators/)
1. Create the noise-free cubes and their masks by inserting 300 random smoothed and resampled mock galaxies randomly into a random noise-free equivalent of the mosaiced cubes.
```bash
src/data_generatores/insert_mock_galaxies_to_noisefree_cubes.py
```
2. Add noise-free cubes to the mosaiced noise cubes:
```bash
src/data_generatores/insert_noisefree_galaxy_cubes_to_mosaics.py
```

### Run SoFiA on cubes
1. Install SoFiA from [here](https://github.com/SoFiA-Admin/SoFiA-2), the installation guide can be found [here](https://github.com/SoFiA-Admin/SoFiA-2/wiki).
2. Make sure to edit the parameter files accordingly for each cube, an explanation of the parameters can be found [here](https://github.com/SoFiA-Admin/SoFiA-2/wiki/SoFiA-2-Control-Parameters).
3. Run sofia on data cubes
```bash
sofia <parameter_file>
```
or if you want to store the time taken:
```bash
usage: src/run_segmentation/run_sofia.py [-h] [--sofia_loc [SOFIA_LOC]] [--cube_dir [CUBE_DIR]] [--param_dir [PARAM_DIR]]

Run SoFiA

optional arguments:
  -h, --help            show this help message and exit
  --sofia_loc [SOFIA_LOC]
                        The sofia executable location (default: /net/blaauw/data2/users/vdhulst/SoFiA-2/sofia)
  --cube_dir [CUBE_DIR]
                        The directory of the cubes (default: ./data/training/loudInput)
  --param_dir [PARAM_DIR]
                        The directory containing the parameter files (default: ./run_segmentation/params)
```
### Run MTObjects
Run MTO with sliding window on all cubes (for memory purposes):
```bash
usage: src/run_segmentation/run_mto.py [-h] [--mto_dir [MTO_DIR]] [--param_file [PARAM_FILE]] [--input_dir [INPUT_DIR]]

Run MTO

optional arguments:
  -h, --help            show this help message and exit
  --mto_dir [MTO_DIR]   The directory of the MTO executable (default: ../mtobjects)
  --param_file [PARAM_FILE]
                        The parameter file (default: ../mtobjects/radio_smoothed-00_F.txt)
  --input_dir [INPUT_DIR]
                        The directory of the input data (default: data/training/loudInput)
```
### Run V-Net
1. Train model on subcubes created from sliding windows, each of dimension 128x128x64.
```bash
usage: src/train_vnet_model.py [-h] [--loaded [LOADED]] [--batch_size [BATCH_SIZE]] [--shuffle [SHUFFLE]] [--num_workers [NUM_WORKERS]] [--dims [DIMS]] [--overlaps [OVERLAPS]] [--root [ROOT]]
                      [--random_seed [RANDOM_SEED]] [--train_size [TRAIN_SIZE]] [--model [MODEL]] [--opt [OPT]] [--lr [LR]] [--inChannels [INCHANNELS]] [--classes [CLASSES]] [--log_dir [LOG_DIR]]
                      [--dataset_name [DATASET_NAME]] [--terminal_show_freq [TERMINAL_SHOW_FREQ]] [--nEpochs [NEPOCHS]] [--scale [SCALE]] [--subsample [SUBSAMPLE]] [--cuda [CUDA]]
                      [--k_folds [K_FOLDS]] [--load_test [LOAD_TEST]]

Train model

optional arguments:
  -h, --help            show this help message and exit
  --batch_size [BATCH_SIZE]
                        Batch size (default: 4)
  --shuffle [SHUFFLE]   Whether or not to shuffle the train/val split (default: True)
  --num_workers [NUM_WORKERS]
                        The number of workers to use (default: 2)
  --dims [DIMS]         The dimensions of the subcubes (default: [128, 128, 64])
  --overlaps [OVERLAPS]
                        The dimensions of the overlap of subcubes (default: [15, 20, 20])
  --root [ROOT]         The root directory of the data (default: ./data/training/)
  --random_seed [RANDOM_SEED]
                        Random Seed (default: 42)
  --train_size [TRAIN_SIZE]
                        Ratio of training to validation split (default: 0.8)
  --model [MODEL]       The 3D segmentation model to use (default: VNET)
  --opt [OPT]           The type of optimizer (default: adam)
  --lr [LR]             The learning rate (default: 0.001)
  --inChannels [INCHANNELS]
                        The desired modalities/channels that you want to use (default: 1)
  --classes [CLASSES]   The number of classes (default: 2)
  --log_dir [LOG_DIR]   The directory to output the logs (default: ./runs/)
  --dataset_name [DATASET_NAME]
                        The name of the dataset (default: hi_source)
  --terminal_show_freq [TERMINAL_SHOW_FREQ]
                        Show when to print progress (default: 500)
  --nEpochs [NEPOCHS]   The number of epochs (default: 10)
  --scale [SCALE]       The scale of inserted galaxies to noise (default: )
  --subsample [SUBSAMPLE]
                        The size of subset to train on (default: 10)
  --cuda [CUDA]         Memory allocation (default: False)
  --k_folds [K_FOLDS]   Number of folds for k folds cross-validations (default: 5)
```
2. Run now trained V-Net on images with sliding window:
```bash
usage: src/run_segmentation/run_vnet.py [-h] [--model [MODEL]] [--opt [OPT]] [--lr [LR]] [--inChannels [INCHANNELS]] [--classes [CLASSES]] [--pretrained [PRETRAINED]] [--test_file [TEST_FILE]]

INFERENCE VNET

optional arguments:
  -h, --help            show this help message and exit
  --model [MODEL]       The 3D segmentation model to use (default: VNET)
  --opt [OPT]           The type of optimizer (default: adam)
  --lr [LR]             The learning rate (default: 0.001)
  --inChannels [INCHANNELS]
                        The desired modalities/channels that you want to use (default: 1)
  --classes [CLASSES]   The number of classes (default: 2)
  --pretrained [PRETRAINED]
                        The location of the pretrained model (default: ./VNET__last_epoch.pth)
  --test_file [TEST_FILE]
                        The file listing the test sliding window pieces (default: ./notebooks/loud_1245mosC-slidingwindowindices.txt)
```
### Investigate Results
1. Create a catalog for the results of each method and cross-reference it with the mask of the inserted galaxies (to mark true or false positives):
```bash
usage: create_catalogs.py [-h] [--data_dir [DATA_DIR]] [--method [METHOD]] [--scale [SCALE]] [--output_dir [OUTPUT_DIR]] [--catalog_loc [CATALOG_LOC]]

Create catalog from output

optional arguments:
  -h, --help            show this help message and exit
  --data_dir [DATA_DIR]
                        The directory containing the data (default: data/)
  --method [METHOD]     The segmentation method being evaluated (default: MTO)
  --scale [SCALE]       The scale of the inserted galaxies (default: loud)
  --output_dir [OUTPUT_DIR]
                        The output directory for the results (default: results/)
```
2. Overlay the false detections on optical images to see if they are real sources:
```bash
usage: overlay_catalog.py [-h] [--method [METHOD]] [--output_file [OUTPUT_FILE]]

Overlay HI moment 0 map on optical cross-matched catalog

optional arguments:
  -h, --help            show this help message and exit
  --method [METHOD]     The method to extract catalogs from (default: SOFIA)
  --output_file [OUTPUT_FILE]
                        The output file for the images (default: ./optical_catalogs/)
```
Manually flag the detected sources that have an tical counterpart by creating a csv with the following columns:
`method,mos_name,label,type`

This also now allows for an evaluation and comparison of all the methods. For this follow the `notebooks/Results.ipynb` notebook. This notebook also includes the addition of shallow machine learning as a post-processing method on all the source finding tools to improve their purity.
### Improve V-Net with newly labelled masks
1. Add the masks of the now deemed real detections (those with optical counterparts) to the ground truth cubes:
```bash
usage: add_masks_of_real_sources_to_ground_truth.py [-h] [--data_dir [DATA_DIR]] [--scale [SCALE]]
                         [--real_cat [REAL_CAT]]

Add real masks to GT

optional arguments:
  -h, --help            show this help message and exit
  --data_dir [DATA_DIR]
                        The directory containing the data (default: ./data/)
  --scale [SCALE]       The scale of the inserted galaxies (default: loud)
  --real_cat [REAL_CAT]
                        The location of file containing flagged real sources (default:
                        ./fp_to_drop.csv)
```
2. Re-train V-Net with newly labelled masks
This can be done in three ways:
- Take ground truth with both real and mock sources and train from scratch
- Take only real sources and continue training VNET from previous point
- Take only real sources and use feature extraction to freeze all the lower layers and only update a new one
