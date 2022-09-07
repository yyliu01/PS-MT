# Getting Started
we visualize our training details via wandb (https://wandb.ai/site).
## visualization
1) you'll need to login
   ```shell 
   $ wandb login
   ```
   you can find you API key in (https://wandb.ai/authorize), and copy & paste it in terminal.
   
2) you can (optionally) add the key to the **main.py** for the server use, with
   ```shell
   import os
   os.environ['WANDB_API_KEY'] = "you key"
   ```
   

## checkpoints
1) for the deeplabv3+ experiments, we utilize exactly same checkpoints as provided by the [CPS](https://github.com/charlesCXK/TorchSemiSeg) in [here](https://1drv.ms/u/s!AsvBenvUFxO3hn9hIs_esxf0aLoH?e=c5cZvF).
2) for the pspnet experiments, we follow the [CCT](https://github.com/yassouali/CCT) as provided in [here](https://1drv.ms/u/s!AsvBenvUFxO3hwCoBdieaWLzV7pp?e=a0C60K).
  
## training
The code is trained under 4xV100(32Gb) for the voc12 dataset, 
and 2xV100 (32Gb) for the cityscapes. 

Our approach performs robust under different hardware's test, please see the training logs for more details.

### VOC12 Setting
**(global)** we utilize batch_size=64 (32 labelled, 32 unlabelled data) for the training, with learning rate 1e-2, under 4 GPUs.

**(local)** in each GPU, we utilize batch_size=16 (8 labelled, 8 unlabelled data) under the learning rate 2.5e-3.

1) augset experiment
   
   | hyper-param 	| 1/16 (662)| 1/8 (1323)| 1/4 (2646)| 1/2 (5291)|
   |:--------:	    |:-----:	|:-----:	|:-----:	|:-----:	|
   | epoch          | 80 	| 80 	| 200 	| 300 	|
   | weight      	| 1.5 	| 1.5 	| 1.5 	| 1.5 	|
   
   run the scripts with 

   ```shell
   # -l -> labelled_num; -g -> gpus; -b -> resnet backbone;
   ./scripts/train_voc_aug.sh -l 1323 -g 2 -b 101
   ```
   
2) high-quality experiment

   | hyper-param 	| 1/16 (92)| 1/8 (183)| 1/4 (366)| 1/2 (732)|
   |:--------:	    |:-----:	|:-----:	|:-----:	|:-----:	|
   | epoch          | 80 	| 80 	| 80 	| 80 	|
   | weight      	| 0.06 	| 0.6 	| 0.6 	| 0.6 	|
   
   run the scripts with 
   ```shell
   # -l -> labelled_num; -g -> gpus; -b -> resnet backbone;
   ./scripts/train_voc_hq.sh -l 732 -g 2 -b 101
   ```
P.S., for 1464 high quality setting, our experiments show that, the training under half of the batch_size (i.e., GPU=2xV100) are likely to perform 
**higher** than the paper reported result. 

### Cityscapes Setting
we utilize batch_size=16 (8 labelled, 8 unlabelled data) under the learning rate 4.5e-3. 
(I have to reduce the batch size, as our dgx faced some issues in 2021 Nov. )

   | hyper-param 	| 1/8 (372) | 1/4 (744) | 1/2 (1488)|
   |:--------:	    |:-----:	|:-----:	|:-----:	|
   | epoch         | 320 	| 450 	| 550 	|
   | weight        | 3.0 	| 3.0	| 3.0 	|

   run the scripts with
   ```shell
   # -l -> labelled_num; -g -> gpus; -b -> resnet backbone;
   ./scripts/train_city.sh -l 372 -g 2 -b 50
   ```
