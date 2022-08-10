import os
import random
import warnings
import argparse
from train import Trainer
from Utils.losses import *
from DataLoader.voc import VOC
import torch.distributed as dist
import torch.multiprocessing as mp
from Utils.tensor_board import Tensorboard
from Model.Deeplabv3_plus.EntireModel import EntireModel as model_deep

from Utils.logger import *
warnings.filterwarnings("ignore")
# from dgx.download_to_pvc import *
# from Model.PSPNet.EntireModel import EntireModel as model_psp


def main(gpu,  ngpus_per_node, config, args):
    args.local_rank = gpu
    if args.local_rank <= 0:
        logger = logging.getLogger("PS-MT")
        logger.propagate = False
        logger.warning("Training start, 总共 {} epochs".format(str(config['trainer']['epochs'])))
        logger.critical("DGX: {} with [{} x v100]".format("On" if args.dgx else "Off", str(args.gpus)))
        logger.critical("GPUs: {}".format(args.gpus))
        logger.critical("Network Architecture: {}, with ResNet {} backbone".format(args.architecture,
                                                                                   str(args.backbone)))
        logger.critical("Current Labeled Example: {}".format(config['n_labeled_examples']))
        logger.critical("Learning rate: other {}, and head is the SAME [world]".format(config['optimizer']['args']['lr']))

        logger.info("Image: {}x{} based on {}x{}".format(config['train_supervised']['crop_size'],
                                                         config['train_unsupervised']['crop_size'],
                                                         config['train_supervised']['base_size'],
                                                         config['train_unsupervised']['base_size']))

        logger.info("Current batch: {} [world]".format(int(config['train_unsupervised']
                                                           ['batch_size']) * args.world_size +
                                                       int(config['train_supervised']
                                                           ['batch_size']) * args.world_size))

        logger.info("Current unsupervised loss function: {}, with weight {} and length {}".format(config['model']['un_loss'],
                                                                                                  config['unsupervised_w'],
                                                                                                  config['ramp_up']))
        logger.info("Current config+args: \n{}".format({**config, **vars(args)}))
    if args.ddp:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=args.local_rank,
            world_size=args.world_size
        )
        if args.gcloud:
            if args.local_rank <= 0:
                logger.info("downloading voc dataset ...")
                download_voc_unzip(data_dir="./VOCtrainval_11-May-2012", prefix="your/prefix/")
                logger.info("downloading resnet {} pretrained checkpoint ...".format(str(args.backbone)))
                download_checkpoint("VocCode/Model/PSPNet/Backbones/pretrained/",
                                    '3x3resnet{}-imagenet.pth'.format(str(args.backbone)))
                
                download_checkpoint("VocCode/Model/Deeplabv3_plus/Backbones/pretrained/",
                                    'resnet{}.pth'.format(str(args.backbone)))
            dist.barrier()

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    # DATA LOADERS
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']

    supervised_loader = VOC(config['train_supervised'], ddp_training=args.ddp, dgx=args.dgx)
    unsupervised_loader = VOC(config['train_unsupervised'], ddp_training=args.ddp, dgx=args.dgx)

    val_loader = VOC(config['val_loader'], dgx=args.dgx)

    iter_per_epoch = len(unsupervised_loader)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    else:
        raise NotImplementedError

    cons_w_unsup = ConsistencyWeight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                     rampup_starts=0, rampup_ends=config['ramp_up'],  ramp_type="cosine_rampup")
    if args.architecture == "psp":
        assert "Code will be uploaded soon."

    elif args.architecture == "deeplabv3+":
        Model = model_deep
        config['model']['data_h_w'] = [config['train_supervised']['crop_size'],
                                       config['train_supervised']['crop_size']]
    else:
        raise NotImplementedError

    model = Model(num_classes=val_loader.dataset.num_classes, config=config['model'],
                  sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
                  weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'],
                  ignore_index=val_loader.dataset.ignore_index)

    if args.local_rank <= 0:
        wandb_run = Tensorboard(config=config, online=False)

    trainer = Trainer(model=model,
                      config=config,
                      supervised_loader=supervised_loader,
                      unsupervised_loader=unsupervised_loader,
                      val_loader=val_loader,
                      iter_per_epoch=iter_per_epoch,
                      train_logger=logger if args.local_rank <= 0 else None,
                      wandb_run=wandb_run if args.local_rank <= 0 else None,
                      args=args)

    trainer.train()
    if args.local_rank <= 0:
        wandb_run.finish()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')

    parser.add_argument('--batch_size', default=8, type=int)
    
    parser.add_argument('--epochs', default=-1, type=int)

    parser.add_argument('--warm_up', default=0, type=int)

    parser.add_argument('--labeled_examples', default=1464, type=int)
    
    parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                        help='Default HEAD Learning rate for PSP, '
                             '*Note: the head layers lr will automatically divide by 10*'
                             '*Note: in ddp training, lr will automatically times by n_gpu')

    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')

    parser.add_argument('--gcloud', action='store_true',
                        help='using gcloud or not.')

    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("-a", "--architecture", default='deeplabv3+', type=str,
                        help="pick a architecture, either pspnet [psp] or deeplabv3+ [deeplabv3+]")

    parser.add_argument("-b", "--backbone", default=50, type=int,
                        help="the backbone is fixed to be resnet -- following previous work;"
                             "the resnet x {50, 101} layers")

    parser.add_argument("--ddp", action="store_true",
                        help="distributed data parallel training or not;"
                             "MUST SPECIFIED")

    parser.add_argument("--dgx", action="store_true",
                        help="use dgx == [4*v100] to train the machine; combo with pvc"
                             "the only difference is the batch v.s. lr")
    
    parser.add_argument('--semi_p_th', type=float, default=0.6,
                        help='positive_threshold for semi-supervised loss')
    
    parser.add_argument('--semi_n_th', type=float, default=0.6,
                        help='negative_threshold for semi-supervised loss')

    parser.add_argument('--unsup_weight', type=float, default=1.5,
                        help='unsupervised weight for the semi-supervised loss')

    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    args.ddp = True if args.gpus > 1 else False

    if args.architecture == "psp":
        config = json.load(open("VocCode/configs/config_psp.json"))
    else:
        config = json.load(open("VocCode/configs/config_deeplab_v3+.json"))

    config['n_gpu'] = args.gpus

    if args.epochs != -1:
        config['trainer']['epochs'] = args.epochs

    config['train_supervised']['batch_size'] = args.batch_size
    config['train_unsupervised']['batch_size'] = args.batch_size
    config['model']['warm_up_epoch'] = args.warm_up
    config['n_labeled_examples'] = args.labeled_examples
    config['model']['resnet'] = args.backbone
    args.world_size = args.gpus * args.nodes
    config['optimizer']['args']['lr'] = args.learning_rate
    config['unsupervised_w'] = args.unsup_weight

    if args.ddp:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9901'
        config['optimizer']['args']['lr'] = config['optimizer']['args']['lr'] * args.world_size
        if args.dgx is True:
            config['train_supervised']['batch_size'] = int(32/args.gpus)
            config['train_unsupervised']['batch_size'] = int(32/args.gpus)

    if args.ddp:
        mp.spawn(main, nprocs=config['n_gpu'], args=(config['n_gpu'], config, args))
    else:
        main(-1, 1, config, args)


