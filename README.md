# Iterative Residual CNNs for Burst Photography Applications (CVPR 2019 submission)

##### Project Website: https://fkokkinos.github.io/deep_burst/

#### Dependencies:

---
This code uses the following external packages:

    Pytorch 1.0
    NumPy
    SciPy
    Matplotlib
    OpenCV
    lmdb

### Models:

---
We provide 3 pre-trained models in the `pretrained_models/` directory. In detail, we provide one model for each task, i.e. burst denoising, burst demosaicking and burst joint denoising-demosaicking.
Beside the models, you will also find the hyper-parameters used to train the models.

### Training:

---
Run the following command to train the  burst denoising model:
```shell
python -B main_burst_denoise.py -depth 5 -epochs 200 -gpu -max_iter 10 -k1 5 -k2 5 -save_path experiment1/ -estimate_noise -init
```
Run the following command to train the burst demosaicking (and denoising) model:
```shell
python -B main_burst_demosaick.py -depth 5 -epochs 200 -gpu -max_iter 10 -k1 5 -k2 5 -save_path experiment1/  -init (-estimate_noise])
```

List of hyper-parameters:
```shell
usage: main.py [-h] -epochs EPOCHS [-depth DEPTH] [-init] [-save_images]
               [-save_path SAVE_PATH] [-gpu] [-num_gpus NUM_GPUS]
               [-max_iter MAX_ITER] -batch_size BATCH_SIZE [-lr LR] [-k1 K1]
               [-k2 K2] [-clip CLIP] [-estimate_noise]


optional arguments:
  -h, --help            show this help message and exit
  -epochs EPOCHS        Number of epochs
  -depth DEPTH          Depth of ResDNet
  -init                 Initialize input with Bilinear Interpolation
  -save_images
  -save_path SAVE_PATH  Path to save model and results
  -gpu
  -num_gpus NUM_GPUS    Note: Currently unused
  -max_iter MAX_ITER    Total number of iterations to use
  -batch_size BATCH_SIZE
  -lr LR
  -k1 K1                Number of iterations to unroll
  -k2 K2                Number of iterations to backpropagate. Use the same
                        value as k1 for TBPTT
  -clip CLIP            Gradient Clip
  -estimate_noise       Estimate noise std via WMAD estimator
```

### Dataset:

---
Download test files from https://gitlab.com/filippakos/burst-cvpr-2019-test-files . Train and validation set used for training of the models are not uploaded due to size restrictions. Send an email to filippos.kokkinos[at]skoltech.ru to request the training sets.  

Each sample returned from a dataloader should be a JSON struct with the following keys:
1. 'image_gt': groundtruth image with shape [C,H,W]
2. 'image_input': burst of B frames with shape [B, C,H,W]
3. 'filename': filename of image or any identifier to be used for image storing
4. 'mask': CFA mask (only for demosaicking)
5. 'warp_matrix': warp matrix that contrains the affine transformations that align the frames according to reference(always assumed to be the last one). The shape is [B,2,3].

---
Bib:
>     @InProceedings{Kokkinos_2019_CVPR,
>               author = {Filippos, Kokkinos and Stamatios, Lefkimmiatis},
>               title = {Iterative Residual CNNs for Burst Photography Applications},
>               booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
>               month = {June},
>               year = {2019}}
