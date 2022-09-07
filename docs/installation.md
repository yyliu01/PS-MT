# Installation
The project is based on the pytorch 1.8.1 with python 3.7.
### 1. Clone the Git  repo
``` shell
$ git clone https://github.com/yyliu01/PS-MT.git
$ cd PS-MT
```
### 2. Install dependencies
1) create conda env
    ```shell
    $ conda env create -f ps-mt.yml
    ```
2) install the torch 1.8.1
    ```shell
    $ conda activate ps-mt
    # IF cuda version < 11.0
    $ pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    # IF cuda version >= 11.0 (e.g., 30x or above)
    $ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

### 3. Prepare dataset

1) please download the VOC12 and Cityscapes dataset (gt_Fine).
2) organize the filepath to point out the root folder for each of them, in the config file.
``` shell
    "train_supervised": {
        "data_dir": "path_to/VOCtrainval_11-May-2012",
        ...
    }
    "train_unsupervised": {
        "data_dir": "path_to/VOCtrainval_11-May-2012",
        ...
    }
    "val": {
        "data_dir": "path_to/VOCtrainval_11-May-2012",
        ...
    }
```
3) (optional) you might need to preprocess Cityscapes labels in [here](https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/preparation),
   as we follow the common setting with **19** classes.   
   
please note, our training details (e.g., training id, input resolution, and batch size) follow the previous works
[CPS](https://github.com/charlesCXK/TorchSemiSeg) and [PseudoSeg](https://github.com/googleinterns/wss).

