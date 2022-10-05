# GGDC
### [Project Page]() | [Paper]()

[![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()<br>

## Updat the script.
4. The outputs can be seen in ```results/males_model/test_latest/traversal/``` or ```results/females_model/test_latest/traversal/``` (according to the se`python download_modelsload the model from ```--which_epoch```. This can be either an epoch number e.g. ```400``` or the latest saved model ```latest```.
2. Test the model: Run```./run_scripts/test.sh``` (Linux) or ```./run_scripts/test.bat``` (windows)
3. The outputs can be seen in ```results/<model name>/test_<model_checkpoint>/index.html```

### Generate Video
1. Prepare a ```.txt``` file with a list of image paths to generate videos for. See examples in ```males_image_list.txt``` and ```females_image_list.txt```
2. Open ```run_scripts/traversal.sh``` (Linux) or ```run_scripts/traversal.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. ```400``` or the latest saved model ```latest```.
   - The relative path to the image list ```--image_path_file```
3. Run ```./run_scripts/traversal.sh``` (Linux) or ```./run_scripts/traversal.bat``` (windows)
4. The output videos will be saved to ```results/<model name>/test_<model_checkpoint>/traversal/```

### Generate Full Progression
This will generate an image of progressions to all anchor classes
1. Prepare a ```.txt``` file with a list of image paths to generate videos for. See examples in ```males_image_list.txt``` and ```females_image_list.txt```
2. Open ```run_scripts/deploy.sh``` (Linux) or ```run_scripts/deploy.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The  name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. ```400``` or the latest saved model ```latest```.
   - The relative path to the image list ```--image_path_file```
3. Run ```./run_scripts/deploy.sh``` (Linux) or ```./run_scripts/deploy.bat``` (windows)
4. The output images will be saved to ```results/<model name>/test_<model_checkpoint>/deploy/```

## Training/Testing on New Datasets
If you wish to train the model on a new dataset, arange it in the following structure:
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
This code is inspired by 
