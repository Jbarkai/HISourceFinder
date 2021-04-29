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

## Usage
### Create Mock Cubes (all scripts in data_generators/)
1. Create the noise-free cubes and their masks by inserting 200-500 random smoothed and resampled mock galaxies randomly into a random noise-free equivalent of the mosaiced cubes.
```bash
usage: make_cubes.py [-h] [--mos_dir [MOS_DIR]] [--gal_dir [GAL_DIR]] [--out_dir [OUT_DIR]] [--cube_file [CUBE_FILE]] [--min_gal [MIN_GAL]] [--max_gal [MAX_GAL]]

Insert mock galaxies into HI cubes

optional arguments:
  -h, --help            show this help message and exit
  --mos_dir [MOS_DIR]   The directory of the noise cubes to insert the mock galaxies into (default: data/mosaics)
  --gal_dir [GAL_DIR]   The directory of the mock galaxy cubes (default: data/mock_gals)
  --out_dir [OUT_DIR]   The output directory of the synthetic cubes (default: data/training)
  --cube_file [CUBE_FILE]
                        The noise cube to insert into (default: data/mosaics/1245mosC.derip.fits)
  --min_gal [MIN_GAL]   The minimum number of galaxies to insert (default: 200)
  --max_gal [MAX_GAL]   The maximum number of galaxies to insert (default: 500)
```
2. Summarize galaxies in resulting noise-free cubes with exploratory plots:
```bash
usage: explore_data.py [-h] [--output_dir [OUTPUT_DIR]] [--root [ROOT]]

Create exploratory plots of noise-free cubes

optional arguments:
  -h, --help            show this help message and exit
  --output_dir [OUTPUT_DIR]
                        Directory to output plots to (default: ../plots/)
  --root [ROOT]         The root directory of the data (default: ../data/training/)
```
3. Scale noise-free cubes and add to noise cubes:
```bash
usage: scale_cubes.py [-h] [--filename [FILENAME]] [--scale [SCALE]]

Scale cubes

optional arguments:
  -h, --help            show this help message and exit
  --filename [FILENAME]
                        Filename (default: loud)
  --scale [SCALE]       Scaling amount (default: noisefree_1245mosB.fits)
```
4. Run data loader to create training and validation sets, each of dimension 128x128x64 made with a sliding window iwth an overlap chosen to include the average size of a galaxy.
```bash
usage: data_loader.py [-h] [--batch_size [BATCH_SIZE]] [--shuffle [SHUFFLE]] [--num_workers [NUM_WORKERS]] [--dims [DIMS]] [--overlaps [OVERLAPS]] [--root [ROOT]]
                      [--random_seed [RANDOM_SEED]] [--scale [SCALE]] [--train_size [TRAIN_SIZE]]

Create training and validation datasets

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
  --root [ROOT]         The root directory of the data (default: ../HISourceFinder/data/training/)
  --random_seed [RANDOM_SEED]
                        Random Seed (default: 42)
  --scale [SCALE]       The scale of inserted galaxies to noise (default: loud)
  --train_size [TRAIN_SIZE]
                        Ratio of training to validation split (default: 0.8)
```
### Train V-Net
Train model on subcubes created from sliding window.
```bash
usage: train_model.py [-h] [--batch_size [BATCH_SIZE]] [--shuffle [SHUFFLE]] [--num_workers [NUM_WORKERS]] [--dims [DIMS]] [--overlaps [OVERLAPS]] [--root [ROOT]]
                      [--random_seed [RANDOM_SEED]] [--train_size [TRAIN_SIZE]] [--model [MODEL]] [--opt [OPT]] [--lr [LR]] [--inChannels [INCHANNELS]]
                      [--inModalities [INMODALITIES]] [--classes [CLASSES]] [--log_dir [LOG_DIR]] [--dataset_name [DATASET_NAME]]
                      [--terminal_show_freq [TERMINAL_SHOW_FREQ]] [--nEpochs [NEPOCHS]] [--scale [SCALE]] [--subsample [SUBSAMPLE]] [--cuda [CUDA]]

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
  --root [ROOT]         The root directory of the data (default: ../HISourceFinder/data/training/)
  --random_seed [RANDOM_SEED]
                        Random Seed (default: 42)
  --train_size [TRAIN_SIZE]
                        Ratio of training to validation split (default: 0.6)
  --model [MODEL]       The 3D segmentation model to use (default: VNET)
  --opt [OPT]           The type of optimizer (default: sgd)
  --lr [LR]             The learning rate (default: 0.001)
  --inChannels [INCHANNELS]
                        The desired modalities/channels that you want to use (default: 1)
  --inModalities [INMODALITIES]
                        The desired number of modalities (default: 1)
  --classes [CLASSES]   The number of classes (default: 1)
  --log_dir [LOG_DIR]   The directory to output the logs (default: ./runs/)
  --dataset_name [DATASET_NAME]
                        The name of the dataset (default: hi_source)
  --terminal_show_freq [TERMINAL_SHOW_FREQ]
                        The maximum number of galaxies to insert (default: 50)
  --nEpochs [NEPOCHS]   The number of epochs (default: 10)
  --scale [SCALE]       The scale of inserted galaxies to noise (default: loud)
  --subsample [SUBSAMPLE]
                        The size of subset to train on (default: 5)
  --cuda [CUDA]         Memory allocation (default: False)
```

## Install and Train LVQ with MTO
Fork and then clone repository.
```bash
git clone https://gitlab.com/michaelvandeweerd/mto-lvq.git
cd mto-lvq
```
Fix out-of-date dependency: replace "GLVQClassifier" with just "GLVQ" in src/lvq.c

Run MTO and LVQ on all files:
```bash
mv ../HISourceFinder/test_mto.sh ./test_mto.sh
bash test_mto.sh
```
MTO usage:
```bash
usage: ./mt-objects <nthreads> <levels> <input> <lambda>
                    [<classifier> [<labels> [<bits> [<mul>
                    [<output> [<3d> [<sigma> [<factor>]]]]]]]]

    nthreads    Numer of threads for sorting and quantizing
    levels      Number of quantized levels and threads for
                refining
    input       Input image
    lambda      
    classifier  LVQ classifier, prompts LVQ segmentation
    labels      Ground truth (FITS) for the input image,
                prompts LVQ training
    bits        Amount of bits per pixel
    mul         Mulfactor bpp(input image) + mulfactor = bits
                per pixel
    output      Output image
    3d          Whether the input image is in 3D
    sigma       Standard deviation
    factor      Move up factor

```

## Compare methods
Use the dice loss to compare the segmentation methods:
```bash
usage: compare_methods.py [-h] [--model [MODEL]] [--opt [OPT]] [--lr [LR]] [--inChannels [INCHANNELS]] [--classes [CLASSES]] [--dataset_name [DATASET_NAME]]
                          [--save [SAVE]] [--pretrained [PRETRAINED]] [--cuda [CUDA]] [--test_dir [TEST_DIR]]

Train model

optional arguments:
  -h, --help            show this help message and exit
  --model [MODEL]       The 3D segmentation model to use (default: VNET)
  --opt [OPT]           The type of optimizer (default: sgd)
  --lr [LR]             The learning rate (default: 0.001)
  --inChannels [INCHANNELS]
                        The desired modalities/channels that you want to use (default: 1)
  --classes [CLASSES]   The number of classes (default: 1)
  --dataset_name [DATASET_NAME]
                        The name of the dataset (default: hi_source)
  --save [SAVE]         The checkpoint location (default: ./inference_checkpoints/VNET_checkpoints/VNET_hi_source)
  --pretrained [PRETRAINED]
                        The checkpoint location (default: ../saved_models/VNET_checkpoints/VNET_/VNET__BEST.pth)
  --cuda [CUDA]         Memory allocation (default: False)
  --test_dir [TEST_DIR]
                        Memory allocation (default: False)
```