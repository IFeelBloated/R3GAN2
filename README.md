## BaselineGAN<br><sub>Official PyTorch implementation of the *placeholder* paper</sub>

We show that a basic GAN ***without any tricks***, using simply a modernized CNN backbone and an improved GAN objective, can beat the SOTA StyleGAN family. Check [our report](https://cs.brown.edu/people/ycheng79/csci1952qs23/Top_Project_1_Nick%20Huang_Jayden%20Yi_Convergence%20of%20Relativistic%20GANs%20With%20Zero-Centered%20Gradient%20Penalties.pdf) on our improved GAN objective, where we address both the non-convergence problem and the mode dropping problem of GAN training.

Our code base is built on top of [stylegan3](https://github.com/NVlabs/stylegan3). However, all of our contributions are self-contained in the BaselineGAN directory:
- BaselineGAN/Networks.py: our ConvNeXt inspired CNN backbone for the generator and the discriminator.
- BaselineGAN/Trainer.py: our gradient-penalized RpGAN objective.

### FFHQ (256×256)

Training command:
```
python train.py --outdir=./training-runs --data=./datasets/ffhq-256x256.zip --gpus=8 --batch=256 --gamma=50 --glr=5e-5 --dlr=5e-5 --mirror=1 --aug=fixed --p=0.15 --tick=1
```

Model checkpoint: https://drive.google.com/file/d/1kFPBSdb7nO9V947-uHm0e_-2ZENEZKpY/view?usp=sharing

| Model | FID | Precision | Recall
|:--:|:--:|:--:|:--:|
| StyleGAN2 | 3.78 | 0.69 | 0.43 |
| StyleGAN3-T | 4.81 | 0.64 | 0.50 |
| StyleGAN3-R | 3.92 | 0.69 | 0.47 |
| Ours | 3.37 | 0.70 | 0.46 |

### CIFAR-10

Training command:
```
python train.py --outdir=./training-runs --data=./datasets/cifar10.zip --gpus=8 --batch=2048 --gamma=0.5 --glr=1e-4 --dlr=1e-4 --mirror=1 --aug=fixed --p=0.5 --tick=1 --cond=1 --preset=cifar
```

*placeholder*