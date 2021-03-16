# Monet Moves

Andy Cai, Stephanie Doan, Michael Inoue, Henry Trinh

## Overview

We use three models for the tasks of art style transfer from Claude Monet's paintings to real photographs and artwork generation from existing artwork:
1. [CycleGAN](https://arxiv.org/abs/1703.10593) based on [Pytorch implementation](https://www.kaggle.com/bootiu/cyclegan-pytorch-lightning/)
2. [Neural style transfer](https://arxiv.org/pdf/1603.08155.pdf) featuring perceptual loss for art style transfer, based on [Keras implementation](https://www.kaggle.com/tarunbisht11/generate-art-using-fast-style-transfer-in-a-second)
3. [DCGAN](https://arxiv.org/abs/1511.06434) based on [Pytorch implmentation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) 

## Setup

### Dataset
Download [Kaggle dataset](https://www.kaggle.com/c/gan-getting-started) of Monet's paintings and real photographs (png files) and put into `/data` folder with the following file structure. 
```
monet-moves
  data
    monet
      monet_
        0a5076d42a.jpg
        0bd913dbc7.jpg
        ...
    photo
      0a0c3a6d07.jpg
      0a0d3e6ea7.jpg
      ...
```

### Virtual Environment

Install and run virtualenv, install dependencies.
```
pip3 install virtualenv
python3 -m virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

Save dependencies and deactivate.
```
pip3 freeze > requirements.txt
deactivate
```
