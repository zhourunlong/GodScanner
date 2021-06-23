# GodScanner
GodScanner （扫描全能神） is a course project for Deep Learning, instructed by Will Wu, IIIS, Tsinghua.

## Usage

1. Download model checkpoint for DE-GAN from https://cloud.tsinghua.edu.cn/f/4d5935b9db2046178403/?dl=1, and extract it under folder `de_gan/`. 

2. Download model checkpoint for DewarpNet from https://cloud.tsinghua.edu.cn/f/be93637d95c64287a276/?dl=1, and extract them under folder `DewarpNet/eval/models/`.

3. Place pictures under any folder you like, and run `python predict.py --img_path=<directory to that folder> --generate`.
