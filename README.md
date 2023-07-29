# Aspect Ratio Bucketer

Iterates over a WebDataset such as cc12m, and saves out new per-bucket webdatasets.

Aim is to try and do work such as resizing and center-cropping on the GPU (we can do this via OpenCV CUDA or torchvision), hopefully even the JPEG decode/encode can be done on-GPU via nvjpeg.

Stretch goal is to use multiple workers (e.g. each thread/process consuming from a different set of webdataset shards).

I provide [instructions for compiling OpenCV with CUDA support](https://gist.github.com/Birch-san/ce02730e7987a7154b6e74efc06f182a).

## Usage

```bash
python -m script.bucket_webdataset.py
```