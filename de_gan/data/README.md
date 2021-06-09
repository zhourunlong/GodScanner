## Download dataset

- NoisyOffice from https://archive.ics.uci.edu/ml/datasets/NoisyOffice
- DIBCO prior 2019 from https://vc.ee.duth.gr/dibco2019/
- DIBCO 2019 from https://vc.ee.duth.gr/dibco2019/benchmark/

## Arrange files

```
│  create_dataset.ipynb
├─DIBCO2009
│  ├─GT
│  └─original
├─DIBCO2010
├─DIBCO2011
├─DIBCO2012
├─DIBCO2013
│  ├─GTimages
│  └─OriginalImages
├─DIBCO2014
├─DIBCO2016
│  ├─DIPCO2016_dataset
│  └─DIPCO2016_Dataset_GT
├─DIBCO2017
│  ├─Dataset
│  └─GT
├─DIBCO2018
│  ├─dataset
│  └─gt
├─DIBCO2019
│  ├─Dataset
│  └─GT
└─SimulatedNoisyOffice
   ├─clean_images_grayscale
   └─simulated_noisy_images_grayscale
```
Files inside each folder are omitted.

## Create dataset



Run all commands in `create_dataset.ipynb`, you will get two folders: `original` and `GT`.
