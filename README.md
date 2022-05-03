# CAFE-GAN-PyTorch

This is An unofficial PyTorch implementation of CAFE-GAN - [Arbitrary Face Attribute Editing with Complementary Attention Feature](https://arxiv.org/pdf/2011.11900v1.pdf)

Inverting 13 attributes respectively. From left to right: _Input, Reconstruction, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young_


## To-Do

- [ ] Upload experimental results
  - [ ] FID
  - [ ] Single Attribute Editing Accuracy
- [ ] Upload Pre-training model

## Requirements

* Python 3.7
* PyTorch 1.7.0
* TensorboardX

```bash
pip3 install -r requirements.txt
```

* Dataset
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
    * [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in `./data/img_align_celeba/*.jpg`
    * [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in `./data/list_attr_celeba.txt`
  * [HD-CelebA](https://github.com/LynnHo/HD-CelebA-Cropper) (optional)
    * Please see [here](https://github.com/LynnHo/HD-CelebA-Cropper).
  * [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset (optional)
    * Please see [here](https://github.com/willylulu/celeba-hq-modified).
    * _Images_ should be placed in `./data/celeba-hq/celeba-*/*.jpg`
    * _Image list_ should be placed in `./data/image_list.txt`
* [Pretrained models](http://bit.ly/attgan-pretrain): we will upload pretrained models to Google Cloud.


## Usage

#### To train an AttGAN on CelebA 128x128

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data CelebA \
--use_stu \
--shortcut_layers 3 \
--data_path path/to/Image_Data \
--attr_path path/to/label_Data \
--img_size 128 \
--batch_size 32 \
--kernel_size 3 \
--num_workers 4 \
--save_interval 5000 \
--sample_interval 1500 \
--data_save_root path/to/data_save_root \
--experiment_name 128_ShortCut3_KernelSize3
```

#### To train an AttGAN on CelebA-HQ 256x256

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data CelebA-HQ \
--n_samples 8 \
--gpu \
--use_stu \
--img_size 256 \
--batch_size 8 \
--epochs 100 \
--shortcut_layers 3 \
--kernel_size 3 \
--one_more_conv \
--data_path path/to/Image_Data \
--attr_path path/to/label_Data \
--image_list_path path/to/Image_List \
--data_save_root path/to/data_save_root \
--experiment_name 256_ShortCut3_KernelSize3
```

#### To visualize training details

```bash
tensorboard \
--logdir path/to/experiment/summary
```

#### To test with single attribute editing

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--use_model 'G' \
--experiment_name path/to/your_experiments \
--data_save_root path/to/data_save_root \
--test_int 0.75 \
--gpu
```

#### To test with attention

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--use_model 'D' \
--experiment_name path/to/your_experiments \
--data_save_root path/to/data_save_root \
--gpu
```

Your custom images are supposed to be in `path/to/Image_data` and you also need an attribute list of the images(.txt) `path/to/label_Data`. Please crop and resize them into square images in advance.

#### Acknowledgement
The code is built upon AttGAN[https://github.com/elvisyjlin/AttGAN-PyTorch], STGAN[https://github.com/bluestyle97/STGAN-pytorch] and ABN[https://github.com/machine-perception-robotics-group/attention_branch_network], thanks for their excellent work!