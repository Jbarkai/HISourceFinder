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
```
Install the required packages.
```bash
cd HISourceFinder
pip3 install -r requirements
```

## Usage
Rescale and smooth each mock galaxy and output the new cubes to `data/mock_gals/smoothed/`:
```bash
usage: prepare_galaxies.py [-h] gal_dir out_dir dim

Rescale and smooth mock galaxies

positional arguments:
  gal_dir     The directory of the mock galaxy cubes
  out_dir     The output directory of the smoothed synthetic cubes
  dim         The dimensions to rescale the galaxies to

optional arguments:
  -h, --help  show this help message and exit
```

Create the simulated cubes by inserting 200-500 random snoothed mock galaxies randomly into a random mosaiced cube.
```bash
usage: make_cubes.py [-h] mos_dir gal_dir out_dir dim no_cubes

Insert mock galaxies into HI cubes

positional arguments:
  mos_dir     The directory of the noise cubes to insert the mock galaxies into
  gal_dir     The directory of the mock galaxy cubes
  out_dir     The output directory of the synthetic cubes
  dim         The dimensions to rescale the galaxies to
  no_cubes    The number of synthetic training cubes to produce

optional arguments:
  -h, --help  show this help message and exit
```
