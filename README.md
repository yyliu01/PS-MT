# PS-MT 
> **[CVPR'22]** [Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/2111.12903)
>
> by Yuyuan Liu, [Yu Tian](https://yutianyt.com/), Yuanhong Chen, [Fengbei Liu](https://fbladl.github.io/), [Vasileios Belagiannis](https://campar.in.tum.de/Main/VasileiosBelagiannis) and [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/)
> 
> Computer Vision and Pattern Recognition Conference (CVPR), 2022

![image](https://user-images.githubusercontent.com/102338056/167279043-362e1405-db45-4355-b92b-0993312fe461.png)


### Installation
Please install the dependencies and dataset based on this [***installation***](./docs/installation.md) document.

### Getting start
Please follow this [***instruction***](./docs/before_start.md) document to reproduce our results.

## Update
* blender setting results in VOC12 dataset (under deeplabv3+ with resnet101)
  
  | Approach  | 1/16 (662)| 1/8 (1323)| 1/4 (2646)| 1/2 (5291)|
  |:--------:	|:-----:	|:-----:	|:-----:	|:-----:	|
  | PS-MT  [(wandb_log)](https://wandb.ai/pyedog1976/blender-exps?workspace=user-pyedog1976)    | 78.79     | 80.29     | 80.66     | 80.87        |
  
  * please note that, we update the blender splits list end with an extra 0 (e.g., 6620 for 662 labels) in [the original directory](https://github.com/yyliu01/PS-MT/tree/main/VocCode/DataLoader/voc_splits). 
  * you can find the related launching scripts in [here](https://github.com/yyliu01/PS-MT/blob/main/scripts/train_voc_blender.sh). 
  * **In case you are using blender experiments (which are built on top of the high-quality labels), please compare with the results in this table**.
  
## Results
### Pascal VOC12 dataset
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
Some examples of training details, including:
1) VOC12 dataset in this [wandb](https://wandb.ai/pyedog1976/PS-MT(VOC12)?workspace=user-pyedog1976) link.
2) Cityscapes dataset in this [wandb](https://wandb.ai/pyedog1976/PS-MT(City)?workspace=user-pyedog1976) link **(w/ 1-teacher inference)**.

In details, after clicking the run (e.g., [1323_voc_rand1](https://wandb.ai/pyedog1976/PS-MT(VOC12)/runs/177s76t6?workspace=user-pyedog1976)), you can checkout:

1) <img src="https://user-images.githubusercontent.com/102338056/167979073-1c1b3144-8a72-4d8d-9084-31d7fdab3e9b.png" width="26" height="22"> overall information (e.g., training command line, hardware information and training time).
2) <img src="https://user-images.githubusercontent.com/102338056/167978940-8c1f3d79-d062-4e7b-b56e-30b97d273ae8.png" width="26" height="22"> training details (e.g., loss curves, validation results and visualization)
3) <img src="https://user-images.githubusercontent.com/102338056/167979238-4847430f-aa0b-483d-b735-8a10b43293a1.png" width="26" height="22"> output logs (well, sometimes might crash ...)

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
- [x] Code of deeplabv3+ for cityscapes
- [ ] Update SOTA methods' results for blender settings.


