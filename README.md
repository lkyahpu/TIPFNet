# TIDFNet
A Transformer-based Infrared Polarization Fusion Network

## Installation
[Python 3.8]
[Pytorch 1.9.0 ]
[cuda 11.4]
[cudnn 8.4.0]

## Dataset：LDDRS
We use the [ LWIR DoFP Dataset of Road Scene (LDDRS)](https://github.com/polwork/LDDRS) as our experimental dataset.
Download LDDRS dataset from https: //github.com/polwork/LDDRS.
You can randomly assign infrared intensity and polarized images for training and testing in the following directories

|-- dataset
  |-- train
    |-- ir
    |-- polar_p
  |-- val
    |-- ir
    |-- polar_p
  |-- test
    |-- ir
    |-- polar_p

## Train & Test
* The pretrained models are provided in the repo. 
* After loading data according to the above directory, you can run `python train.py` and `python test.py` for training and testing respectively.
