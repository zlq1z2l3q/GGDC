# Lifespan Age Synthesis with Geometry Guidance and Decorrelation Constraint
### [Project Page]() | [Paper]()

[![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()<br>

[-]()<sup>1</sup> ,
[-]()<sup>1</sup>,
[-]()<sup>2</sup>,
[-]()<sup>3</sup>,
[-]()<sup>1</sup><br>
<sup>1</sup>-, <sup>2</sup>-, <sup>3</sup>-

<div align="center">

</div>

## Overview
-<br>
This code is the official PyTorch implementation of the paper:
> **Lifespan Age Synthesis with Geometry Guidance and Decorrelation Constraint**<br>
> -, -, -, -, -<br>
> ECCV 2020<br>
> https://arxiv.org/pdf/2003.09764.pdf

## Preparation

Please follow [this github](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis) to prepare the environments and dataset.

## Training and Testing (link to the pretrained models in the colab)
Download the dataset from [here]() and put them under the `datasets` folder.<br>
Training (please modify `--dataroot`, `--name`):
```
sh train_GGDC.sh
```
Testing (please modify `--dataroot`, `--name`, `--which_epoch`, and `--checkpoing_dir`):
```
sh test_GGDC.sh
```

## Quick Demo
You can run the demo locally or explore it in Colab [![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()<br>

## Training/Testing on New Datasets
If you wish to train the model on a new dataset, arrange it in the following structure:
```                                                                                           
├── dataset_name                                                                                                                                                                                                       
│   ├── train<class1> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ...         
│   │   ├── landmarks
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ...                                                                                                                          
...
│   ├── train<classN> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ... 
│   │   ├── landmarks
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ...   
│   ├── test<class1> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ...                                                                                                                             
...
│   ├── test<classN> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ... 
``` 

## Citation
If you use this code for your research, please cite our paper.
```
```

## Acknowledgments
This code is inspired by [LATS](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis) and [DLFS](https://github.com/SenHe/DLFS).
