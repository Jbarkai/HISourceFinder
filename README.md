# A comparative study of source-finding techniques in HI emission line cubes using SoFiA, MTObjects, and supervised deep learning

<p align="center">
  <img src="https://github.com/Jbarkai/HISourceFinder/blob/master/cover_pic.png" width="500" />
</p>

Corresponding Authors: J.A. Barkai, M.A.W. Verheijen, E.T. Martínez, M.H.F. Wilkinson

The published paper produced using this code can be found [here](https://www.aanda.org/component/article?access=doi&doi=10.1051/0004-6361/202244708).
Please cite this linked work as follows:
J.A. Barkai, M.A.W. Verheijen, E.T. Martínez, M.H.F. Wilkinson (2022), ‘A comparative study of source-finding techniques
in HI emission line cubes using SoFiA, MTObjects, and supervised deep learning’, Astronomy & Astrophysics.

## Project Description
**Context**: The 21 cm spectral line emission of atomic neutral hydrogen (HI) is one of the primary wavelengths observed in radio
astronomy. However, the signal is intrinsically faint and the HI content of galaxies depends on the cosmic environment, requiring
large survey volumes and survey depth to investigate the HI Universe. As the amount of data coming from these surveys continues to
increase with technological improvements, so does the need for automatic techniques for identifying and characterising HI sources
while considering the tradeoff between completeness and purity.

**Aims**: This study aimed to find the optimal pipeline for finding and masking the most sources with the best mask quality and the fewest
artefacts in 3D neutral hydrogen cubes. Various existing methods were explored, including the traditional statistical approaches and
machine learning techniques, in an attempt to create a pipeline to optimally identify and mask the sources in 3D neutral hydrogen
(HI) 21 cm spectral line data cubes.

**Methods**: Two traditional source-finding methods were tested first: the well-established HI source-finding software SoFiA and one
of the most recent, best performing optical source-finding pieces of software, MTObjects. A new supervised deep learning approach
was also tested, in which a 3D convolutional neural network architecture, known as V-Net, which was originally designed for medical
imaging, was used. These three source-finding methods were further improved by adding a classical machine learning classifier as a
post-processing step to remove false positive detections. The pipelines were tested on HI data cubes from the Westerbork Synthesis
Radio Telescope with additional inserted mock galaxies.

**Results**: Following what has been learned from work in other fields, such as medical imaging, it was expected that the best pipeline
would involve the V-Net network combined with a random forest classifier. This, however, was not the case: SoFiA combined with a
random forest classifier provided the best results, with the V-Net–random forest combination a close second. We suspect this is due to
the fact that there are many more mock sources in the training set than real sources. There is, therefore, room to improve the quality
of the V-Net network with better-labelled data such that it can potentially outperform SoFiA

## Quickstart

### Setup
1. Clone the repository.
```bash
git clone https://github.com/Jbarkai/HISourceFinder.git
cd HISourceFinder
```
2. Create and activate a Python 3.7 environment.
```bash
conda create -n hisources python=3.7
source activate hisources
```
3. Install the required packages.
```bash
pip --cache-dir /tmp/pipcache install -r requirements.txt
```

### Requirements
- A library of mock galaxy fits files
- A library of HI emission cube fits files
- A library of noise normalised HI emission cube fits files
- An empty directory for each of the following:
  - The noise free files containing mock galaxies
  - The HI emission cubes containing the mock galaxies
  - The ground truth mask for the mock galaxies

### Inserting the mock galaxies into the HI emisison cubes
1. Create the noise-free cubes and their masks by inserting n random smoothed and resampled mock galaxies randomly into a noise-free equivalent of the HI emission cubes.
```bash
usage: src/data_generatores/insert_mock_galaxies_to_noisefree_cubes.py [-h] [--gal_dir] [--out_dir] [--cube_file] [--no_gals]

optional arguments:
  -h, --help            show this help message and exit
  --gal_dir
    The directory of the mock galaxy cubes (default: data/mock_gals)
  --out_dir
    The output directory of the synthetic cubes (default: data/training)
  --cube_file
    The HI emission cube to insert into (default: data/mosaics/1245mosC.derip.norm.fits)
  --no_gals
    The number of galaxies to insert (default: 300)
```
2. Add the noise-free cubes to the HI emission cubes:
```bash
usage: src/data_generatores/insert_noisefree_galaxy_cubes_to_mosaics.py [-h] [--noise_free_file] [--orig_file] [--noise_file] [--out_dir]

optional arguments:
  -h, --help            show this help message and exit
  --noise_free_file
    The file name of the noise free cube with inserted galaxies (default: ./data/training/Input/noisefree_1245mosC.fits)
  --orig_file
    The file name of the original, un-normalised HI emission cube (default: ./data/orig_mosaics/1245mosC.derip.fits)
  --noise_file
    The file name of the normalised HI emission cube (default: ./data/mosaics/1245mosC.derip.norm.fits)
  --out_dir
    The output directory of the created cubes (default: data/training/)
```

### Train V-Net
V-Net is originally designed for locating and masking objects in medical images (Milletari et al. 2016) and was chosen due to its ability to take full
data volumes as its input as opposed to slices of 3D images. VNet is a fully volumetric CNN built following the well-known
architecture of U-Net (Christ et al. 2016).

The code for V-Net was adapted from [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)(Nikolaos 2019),
an open-source 3D medical segmentation library. Access to the trained V-Net model (with weights) can be requested by
emailing: jordan.barkai@gmail.com

The model needs to be trained on sub-cubes created from a sliding window, each of dimension 128x128x64.
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

### Run the all the source finders on the cubes
#### SoFiA
SoFiA (Serra et al. 2015; Westmeier et al. 2021). SoFiA is designed to be independent of the source of HI
emission line data used and is currently the most used pipeline for source-finding in HI emission cubes.

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
#### MTObjects
MTObjects (Teeninga et al. 2013, 2016) is region-based source-finding software that makes use of max-trees.
This software was originally designed for 2D optical data. However, since diffuse optical sources similarly
suffer from the tradeoff between completeness and purity, MTO has been extended further for HI emission cubes
by Arnoldus (2015). Access to this version can be requested by emailing: m.h.f.wilkinson@rug.nl

To run MTO using a sliding window on all cubes, storing the time taken:
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
#### V-Net
Once trained, the V-Net model can be ran on the HI emission cubes with a sliding window:
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
A catalog for the results of each method can be created and cross-referenceed with the mask of the inserted galaxies (to mark true or false positives):
```bash
usage: create_catalogs.py [-h] [--data_dir [DATA_DIR]] [--method [METHOD]] [--output_dir [OUTPUT_DIR]] [--catalog_loc [CATALOG_LOC]]

Create catalog from output

optional arguments:
  -h, --help            show this help message and exit
  --data_dir [DATA_DIR]
                        The directory containing the data (default: data/)
  --method [METHOD]     The segmentation method being evaluated (default: SOFIA)
  --output_dir [OUTPUT_DIR]
                        The output directory for the results (default: results/)
```
Overlay the false detections on optical images, taken from the Panoramic Survey Telescope and Rapid Response System
(Pan-STARRS) survey (Flewelling et al. 2020), to see if they are real sources:
```bash
usage: overlay_catalog.py [-h] [--method [METHOD]] [--output_file [OUTPUT_FILE]] [--catalogue_dir [CATALOGU_DIR]]

Overlay HI moment 0 map on optical cross-matched catalog

optional arguments:
  -h, --help            show this help message and exit
  --method [METHOD]     The method to extract catalogs from (default: SOFIA)
  --output_file [OUTPUT_FILE]
                        The output file for the images (default: ./optical_catalogs/)
  --catalogue_dir [CATALOGU_DIR]
                        The file containing the catalogue of found sources (default: ./results/)
```

### References
Arnoldus, C. (2015), A max-tree-based astronomical source finder, Master’s thesis, University of Groningen.

Christ, P. et al. (2016), Automatic liver and lesion segmentation in CT using cascaded fully convolutional neural networks and 3D conditional random fields,
in S. Ourselin, L. Joskowicz, M. Sabuncu, G. Unal and W. Wells, eds, ‘Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016’,
Springer International Publishing, Cham, pp. 415–423.

Flewelling, H. et al. (2020), ‘The Pan-STARRS1 database and data products’,
The Astrophysical Journal Supplement Series 251(1), 7.

Milletari, F., Navab, N. and Ahmadi, S. (2016), ‘V-Net: Fully convolutional neural networks for volumetric medical image segmentation’, Proceedings - 2016
4th International Conference on 3D Vision pp. 565–571.

Nikolaos, A. (2019), Deep learning in medical image analysis: a comparative
analysis of multi-modal brain-MRI segmentation with 3D deep neural networks, Master’s thesis, University of Patras.

Serra, P. et al. (2015), ‘SoFiA: A flexible source finder for 3D spectral line data’,
Monthly Notices of the Royal Astronomical Society 448(2), 1922–1929.

Teeninga, P., Moschini, U., Trager, S. C. and Wilkinson, M. H. F. (2013), Bivariate statistical attribute filtering: A tool for robust detection of faint objects,
in ‘11th International Conference "Pattern Recognition and Image Analysis:
New Information Technologies" (PRIA-11-2013)’, pp. 746–749

Westmeier, T., Kitaeff, S., Pallot, D., Serra, P., van der Hulst, J. M., Jurek, R. J.,
Elagali, A., For, B. Q., Kleiner, D., Koribalski, B. S., Lee-Waddell, K., Mould,
J. R., Reynolds, T. N., Rhee, J. and Staveley-Smith, L. (2021), ‘SOFIA 2 - An
automated, parallel HI source finding pipeline for the WALLABY survey’,
MNRAS 506(3), 3962–3976