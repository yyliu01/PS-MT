# PS-MT 
> [cvpr22] [Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/2111.12903)
>
> by Yuyuan Liu, Yu Tian, Yuanhong Chen, Fengbei Liu, Vasileios Belagiannis, Gustavo Carneiro
> 
> Computer Vision and Pattern Recognition Conference (CVPR), 2022

![image](https://user-images.githubusercontent.com/102338056/167279043-362e1405-db45-4355-b92b-0993312fe461.png)


### Installation
Please install the dependencies and dataset based on this [***installation***](./docs/installation.md) document.

### Getting start
Please follow this [***instruction***](./docs/before_start.md) document to reproduce our results.

## Results
### Pascal VOC 12 dataset
1. augmented set 

    | Backbone 	| 1/16 (662)| 1/8 (1323)| 1/4 (2646)| 1/2 (5291)|
    |:--------:	|:-----:	|:-----:	|:-----:	|:-----:	|
    | 50       	| 72.83 	| 75.70 	| 76.43 	| 77.88 	|
    | 101      	| 75.50 	| 78.20 	| 78.72 	| 79.76 	|
2. high quality set (based on res101)

   | 1/16 (92)| 1/8 (183)| 1/4 (366)| 1/2 (732)| full (1464)|
   |:-----:	|:-----:	|:-----:	|:-----:	|:-----:	|
   | 65.80 	| 69.58 	| 76.57 	| 78.42 	|80.01|

### CityScape dataset
1. following the setting of [CAC](https://arxiv.org/pdf/2106.14133.pdf) (720x720, CE supervised loss)
   
    | Backbone 	| slid. eval| 1/8 (372)| 1/4 (744)| 1/2 (1488)|
    |:--------:	|:-----:	|:-----:	|:-----:	|:-----:	|
    | 50       	| ✗	        |74.37 	    | 75.15 	| 76.02 	| 
    | 50       	| ✓	        |75.76 	    | 76.92 	| 77.64 	| 
    | 101      	| ✓	        |76.89	    | 77.60 	| 79.09 	|       
2. following the setting of [CPS](https://arxiv.org/pdf/2106.01226.pdf) (800x800, OHEM supervised loss)
   
   | Backbone 	| slid. eval| 1/8 (372)| 1/4 (744)| 1/2 (1488)|
   |:--------:	|:-----:	|:-----:	|:-----:	|:-----:	|
   | 50       	| ✓		    |77.12 	    | 78.38 	| 79.22 	|


## Training details
Some example training details of VOC dataset can be visualized in this [wandb](https://wandb.ai/pyedog1976/PS-MT(VOC12)?workspace=user-pyedog1976) link.

In details, after clicking the run, you can checkout:

1) <img src="https://user-images.githubusercontent.com/102338056/167282405-a116af68-c5c0-46aa-8cc5-1fe2774edf0b.png" width="25" height="25"> overall information (e.g., training command line, hardware information and training time).
2) <img src="https://user-images.githubusercontent.com/102338056/167282511-fd9e21be-9958-41fd-ac9d-b0b9814693b4.png" width="25" height="25"> training details (e.g., loss curves, validation results and visualization)
3) <img src="https://user-images.githubusercontent.com/102338056/167282559-2e5b3d12-0d06-411b-8a23-5be810876309.png" width="25" height="25"> output logs (well, sometimes might crash ...)

## Acknowledgement & Citation
The code is highly based on the [CCT](https://github.com/yassouali/CCT). Many thanks for their great work.

Please consider citing this project in your publications if it helps your research.
```bibtex
@article{liu2021perturbed,
  title={Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation},
  author={Liu, Yuyuan and Tian, Yu and Chen, Yuanhong and Liu, Fengbei and Belagiannis, Vasileios and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2111.12903},
  year={2021}
}

```

#### TODO
- [x] Code of deeplabv3+ for voc12
- [ ] Code of deeplabv3+ for cityscapes
- [ ] Code of pspnet for voc12


