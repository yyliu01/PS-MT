# Getting Started
we visualize our training details via wandb (https://wandb.ai/site).
## visualization
1) you'll need to login
   ```shell 
   $ wandb login
   ```
2) you'll need to copy & paste you API key in terminal
   ```shell
   $ https://wandb.ai/authorize
   ```
   or add the key to the **main.py** with
   ```shell
   import os
   os.environ['WANDB_API_KEY'] = "you key"
   ```
## training
The code is trained under 4xV100(32Gb) for the voc12 dataset, 
and 2xV100 (32Gb) for the cityscapes. 

Please note, our approach performs robust under different hardware's test. See the training logs for more details.

### voc12 setting
**(global)** we utilize batch_size=64 (32 labelled, 32 unlabelled data) for the training, with learning rate 1e-2, under 4 GPUs.

**(local)** in each GPU, we utilize batch_size=16 (8 labelled, 8 unlabelled data) under the learning rate 2.5e-3.

1) augset experiment
   
   | hyper-param 	| 1/16 (662)| 1/8 (1323)| 1/4 (2646)| 1/2 (5291)|
   |:--------:	    |:-----:	|:-----:	|:-----:	|:-----:	|
   | epoch          | 80 	| 80 	| 200 	| 300 	|
   | weight      	| 1.5 	| 1.5 	| 1.5 	| 1.5 	|

2) high-quality experiment

   | hyper-param 	| 1/16 (92)| 1/8 (183)| 1/4 (366)| 1/2 (732)|
   |:--------:	    |:-----:	|:-----:	|:-----:	|:-----:	|
   | epoch          | 80 	| 80 	| 80 	| 80 	|
   | weight      	| 0.06 	| 0.6 	| 0.6 	| 0.6 	|

P.S., our experiments show that, the training under half of the batch_size (i.e., GPU=2xV100) are likely to perform 
**higher** than the paper reported result. 

### Cityscapes setting
we utilize batch_size=16 (8 labelled, 8 unlabelled data) under the learning rate 4.5e-3. 
(I have to reduce the batch size, as our dgx faced some issues in 2021 Nov. )

| hyper-param 	| 1/8 (372) | 1/4 (744) | 1/2 (1488)|
|:--------:	    |:-----:	|:-----:	|:-----:	|
| epoch         | 320 	| 450 	| 550 	|
| weight        | 3.0 	| 3.0	| 3.0 	|

