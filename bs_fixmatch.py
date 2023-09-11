#import needed library
import os
import logging
import random
import warnings

from utils import net_builder, get_logger, count_parameters, time_str
from utils import TBLog, get_SGD, get_cosine_schedule_with_warmup
from models.fixmatch.fixmatch import FixMatch
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader
from config.config import parse_commandline_args, save_config_yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse





config_file = "./config/fixmatch.yaml"  # 

def main(args):
    # set the output path
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_name is None or args.save_name == "":
        args.save_name = "{}_{}_s{}".format(args.dataset, args.num_labels, args.seed)
    save_path = os.path.join(args.save_dir, args.save_name)

    # if os.path.exists(save_path) and not args.overwrite:  # solve by time string
    #     raise Exception('already existing model: {}'.format(save_path))

    if args.resume:
        if args.load_path is None or args.load_path == "":
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        # args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    #distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node
    
    #divide the batch_size according to the number of nodes
    args.batch_size = int(args.batch_size / args.world_size)

    # gol._init()
    # gol.set_value('iter', 0)
    
    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size 
        
        #args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) 
    else:
        main_worker(args.gpu, ngpus_per_node, args)


    

def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''
    
    global best_acc1
    args.gpu = gpu
    
    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu # compute global rank
        
        # set distributed group:
        print(args.dist_backend, args.dist_url, args.world_size, args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    ###########################
    # 1. output settings
    ###########################
    #SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    name_by_time = time_str()
    
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, name_by_time)
        logger_level = "INFO"
    logger = get_logger(args.save_name, save_path, f'{name_by_time}.log',logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    csv_path = os.path.join(save_path, "{}_stat.csv".format(name_by_time))
    
    ###########################
    # 2. Initialize Model
    ###########################
    # SET FixMatch: class FixMatch in models.fixmatch
    args.bn_momentum = 1.0 - args.ema_m
    _net_builder = net_builder(args.net, 
                               args.net_from_name,
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'bn_momentum': args.bn_momentum,
                                'dropRate': args.dropout})
    
    model = FixMatch(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     not args.soft_label,
                     num_eval_iter=args.num_eval_iter,
                     tb_log=tb_log,
                     logger=logger, 
                     csv_path=csv_path)

    logger.info('Number of Trainable Params: {:.2f}M.'.format(count_parameters(model.train_model) /1e6))

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter*0)
    ## set SGD and cosine lr on FixMatch 
    model.set_optimizer(optimizer, scheduler)
    
    ###########################
    # 3. Put Model onto GPUs
    ###########################
    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            
            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.train_model.cuda(args.gpu)
            model.train_model = torch.nn.parallel.DistributedDataParallel(model.train_model,
                                                                          device_ids=[args.gpu])
            model.eval_model.cuda(args.gpu)
            
        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.train_model = model.train_model.cuda(args.gpu)
        model.eval_model = model.eval_model.cuda(args.gpu)
        
    else:
        model.train_model = torch.nn.DataParallel(model.train_model).cuda()
        model.eval_model = torch.nn.DataParallel(model.eval_model).cuda()
    
    # logger.info(f"model_arch: {model.train_model}")
    logger.info(f"Arguments: {args}")
    
    # cudnn.benchmark = True
    cudnn.benchmark = False


    ###########################
    # 4. Prepare DataLoader
    ###########################
    # Construct Dataset & DataLoader
    train_dset = SSL_Dataset(name=args.dataset, train=True, nfold=args.nfold,
                             num_classes=args.num_classes, data_dir=args.data_dir)
    lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)
    
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, 
                             num_classes=args.num_classes, data_dir=args.data_dir)
    eval_dset = _eval_dset.get_dset()
    
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}
    # # dset,
    # batch_size = None,
    # shuffle = False,
    # num_workers = 4,
    # pin_memory = True,
    # data_sampler = None,
    # replacement = True,
    # num_epochs = None,
    # num_iters = None,
    # generator = None,
    # drop_last=True,
    # distributed=False):
    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler = args.train_sampler,
                                              num_iters=256,
                                              num_workers=4*args.num_workers,
                                              distributed=args.distributed, 
                                              pin_memory=args.pin_memory)
    
    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size*args.uratio,
                                               data_sampler = args.train_sampler,
                                               num_iters=256,
                                               num_workers=4*args.num_workers,
                                               distributed=args.distributed,
                                               pin_memory=args.pin_memory)
    
    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size, 
                                          num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    
    ## set DataLoader on FixMatch
    model.set_data_loader(loader_dict)
    

    ###########################
    # 5. Resume/Start training 
    ###########################
    #If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)
    
    # START TRAINING of FixMatch
    trainer = model.train
    logger.info("========================= Start Training =========================")
    logger.info(f"  Data   >> M:{args.net}. Dataset:{args.dataset}. labeledNum:{args.num_labels}, Seed:{args.seed}")
    logger.info(f"  Train  >> Iters:{args.num_train_iter}. Batch:{args.batch_size}/{args.uratio} LR:{args.lr} Mom:{args.momentum}")
    logger.info(f"  Pseudo >> Soft:{args.soft_label}. Thred:{args.p_cutoff}. Sharpen:{args.T}. lossW:{args.ulb_loss_ratio}.")
    logger.info(f"  Debug  >> TBD....")
    logger.info("-"*66)
    for epoch in range(args.epoch):
        trainer(args, logger=logger)
        
    if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)
        
    logging.warning(f"GPU {args.rank} training is FINISHED")
    

if __name__ == "__main__":
    # config_file = None
    description = "Train DaSSL"
    my_args = parse_commandline_args(name=description, filepath=config_file)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--label', action='store', type=int, required=True)
    # parser.add_argument('--seed', action='store', type=int, required=True)
    # args = parser.parse_args()

    # my_args.num_labels = args.label
    # my_args.seed = args.seed

    main(my_args)
