import sys
import os
import argparse
import random
import shutil
import time
import warnings
import json

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import numpy as np
from thop import profile
from thop import clever_format
import apex
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from config import Config
from public.detection.dataset.cocodataset import COCODataPrefetcher, collater,collaterdata
from public.detection.models.loss import RetinaLoss
from public.detection.models.decode import RetinaDecoder
from public.detection.models import retinanet
from public.imagenet.utils import get_logger
from pycocotools.cocoeval import COCOeval
from geomloss import SamplesLoss
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch COCO Detection Training')
    parser.add_argument('--network',
                        type=str,
                        default=Config.network,
                        help='name of network')
    parser.add_argument('--network_t',
                        type=str,
                        default=Config.network_t,
                        help='name of network')
    parser.add_argument('--network_s',
                        type=str,
                        default=Config.network_s,
                        help='name of network')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--lr_t',
                        type=float,
                        default=Config.lr_t,
                        help='learning rate')
    parser.add_argument('--lr_s',
                        type=float,
                        default=Config.lr_s,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--per_node_batch_size',
                        type=int,
                        default=Config.per_node_batch_size,
                        help='per_node batch size')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='load pretrained model params or not')
    parser.add_argument('--pretrained_t',
                        type=bool,
                        default=Config.pretrained_t,
                        help='load pretrained model params or not')
    parser.add_argument('--pretrained_s',
                        type=bool,
                        default=Config.pretrained_s,
                        help='load pretrained model params or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=Config.input_image_size,
                        help='input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--resume_t',
                        type=str,
                        default=Config.resume_t,
                        help='put the path to resuming file if needed')
    parser.add_argument('--resume_s',
                        type=str,
                        default=Config.resume_s,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate_t',
                        type=str,
                        default=Config.evaluate_t,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')
    parser.add_argument('--sync_bn',
                        type=bool,
                        default=Config.sync_bn,
                        help='use sync bn or not')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='LOCAL_PROCESS_RANK')



    return parser.parse_args()

def kl_loss(pre1, pre2):
    criterion_softmax = torch.nn.Softmax(dim=1).cuda()
    pre1 = criterion_softmax(pre1)
    pre2 = criterion_softmax(pre2)
    loss = torch.mean(torch.sum(pre2 * torch.log(1e-8 + pre2 / (pre1 + 1e-8)), 1))
    return loss


def validate(val_dataset, model, decoder, args):
    model = model.module
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        all_eval_result = evaluate_coco(val_dataset, model, decoder, args)

    return all_eval_result


def evaluate_coco(val_dataset, model, decoder, args):

    results, image_ids = [], []
    indexes = []
    for index in range(len(val_dataset)):
        indexes.append(index)

    batch_size = args.per_node_batch_size
    eval_collater = collaterdata()
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=eval_collater.next)

    start_time = time.time()

    for i, data in tqdm(enumerate(val_loader)):
        images, scales = torch.tensor(data['img']), torch.tensor(data['scale'])
        per_batch_indexes = indexes[i * batch_size:(i + 1) * batch_size]

        images = images.cuda().float()
        cls_heads, reg_heads, batch_anchors,C6 = model(images)
        scores, classes, boxes = decoder(cls_heads, reg_heads, batch_anchors)

        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        scales = scales.unsqueeze(-1).unsqueeze(-1)
        boxes /= scales

        for per_image_scores, per_image_classes, per_image_boxes, index in zip(
                scores, classes, boxes, per_batch_indexes):
            # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
            per_image_boxes[:, 2:] -= per_image_boxes[:, :2]

            for object_score, object_class, object_box in zip(
                    per_image_scores, per_image_classes, per_image_boxes):
                object_score = float(object_score)
                object_class = int(object_class)
                object_box = object_box.tolist()
                if object_class == -1:
                    break

                image_result = {
                    'image_id':
                        val_dataset.image_ids[index],
                    'category_id':
                        val_dataset.find_category_id_from_coco_label(object_class),
                    'score':
                        object_score,
                    'bbox':
                        object_box,
                }
                results.append(image_result)

            image_ids.append(val_dataset.image_ids[index])

            print('{}/{}'.format(index, len(val_dataset)), end='\r')

    testing_time = (time.time() - start_time)
    per_image_testing_time = testing_time / len(val_dataset)

    print(f"per_image_testing_time:{per_image_testing_time:.3f}")

    if not len(results):
        print(f"No target detected in test set images")
        return

    json.dump(results,
              open('{}_bbox_results.json'.format(val_dataset.set_name), 'w'),
              indent=4)

    # load results in COCO evaluation tool

    coco_true = val_dataset.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(
        val_dataset.set_name))

    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    all_eval_result = coco_eval.stats

    return all_eval_result

def main():
    args = parse_args()
    global local_rank
    local_rank = args.local_rank
    global logger
    logger = get_logger(__name__, args.log)


    torch.cuda.empty_cache()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    torch.cuda.set_device(local_rank)
    #dist.init_process_group(backend='nccl', init_method='env://')
    global gpus_num
    gpus_num = torch.cuda.device_count()
    if local_rank == 0:
        logger.info(f'use {gpus_num} gpus')
        logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    if local_rank == 0:
        logger.info('start loading data')

    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.per_node_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=collater)
    if local_rank == 0:
        logger.info('finish loading data')

    model_t = retinanet.__dict__[args.network_t](**{
        "pretrained": args.pretrained_t,
        "num_classes": args.num_classes,
    })
    model_s = retinanet.__dict__[args.network_s](**{
        "pretrained": args.pretrained_s,
        "num_classes": args.num_classes,
    })

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)


    flops_t, params_t = profile(model_t, inputs=(flops_input, ))
    flops_t, params_t = clever_format([flops_t, params_t], "%.3f")
    logger.info(f"model: '{args.network_t}', flops: {flops_t}, params: {params_t}")

    flops_s, params_s = profile(model_s, inputs=(flops_input,))
    flops_s, params_s = clever_format([flops_s, params_s], "%.3f")
    logger.info(f"model: '{args.network_s}', flops: {flops_s}, params: {params_s}")

    criterion_t = RetinaLoss(image_w=args.input_image_size,
                             image_h=args.input_image_size).cuda()
    decoder_t = RetinaDecoder(image_w=args.input_image_size,
                              image_h=args.input_image_size).cuda()

    criterion_s = RetinaLoss(image_w=args.input_image_size,
                             image_h=args.input_image_size).cuda()
    decoder_s = RetinaDecoder(image_w=args.input_image_size,
                              image_h=args.input_image_size).cuda()

    emdloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05).cuda()

    model_t = model_t.cuda()
    model_s = model_s.cuda()
    optimizer_t = torch.optim.AdamW(model_t.parameters(), lr=args.lr_t)
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t,
                                                             patience=3,
                                                             verbose=True)
    optimizer_s = torch.optim.AdamW(model_s.parameters(), lr=args.lr_s)
    scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_s,
                                                             patience=3,
                                                             verbose=True)

    if args.sync_bn:
        model_t = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
        model_s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_s)

    if args.apex:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        model_t, optimizer_t = amp.initialize(model_t, optimizer_t, opt_level='O0')
        model_t = apex.parallel.DistributedDataParallel(model_t,
                                                        delay_allreduce=True)

        model_s, optimizer_s = amp.initialize(model_s, optimizer_s, opt_level='O0')
        model_s = apex.parallel.DistributedDataParallel(model_s,
                                                        delay_allreduce=True)
        if args.sync_bn:
            model_t = apex.parallel.convert_syncbn_model(model_t)
            model_s = apex.parallel.convert_syncbn_model(model_s)
    else:
        print(local_rank)
        model_t=nn.parallel.DataParallel(model_t)

        model_s = nn.parallel.DataParallel(model_s)
        """
        model_t = nn.parallel.DistributedDataParallel(model_t,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

        model_s = nn.parallel.DistributedDataParallel(model_s,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
        """


    if args.evaluate_t and args.evaluate_s:
        if not os.path.isfile(args.evaluate):
            raise Exception(
                f"{args.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.evaluate}")
        checkpoint_t = torch.load(args.evaluate_t,
                                  map_location=torch.device('cpu'))
        checkpoint_s = torch.load(args.evaluate_s,
                                  map_location=torch.device('cpu'))
        model_t.load_state_dict(checkpoint_t['model_state_dict'])
        model_s.load_state_dict(checkpoint_s['model_state_dict'])
        logger.info(f"start eval.")
        all_eval_result_t = validate(Config.val_dataset, model_t, decoder_t,args)
        all_eval_result_s = validate(Config.val_dataset, model_s, decoder_s,args)
        logger.info(f"eval done.")
        if all_eval_result_t is not None:
            logger.info(
                f"val: epoch: {checkpoint_t['epoch']:0>5d}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result_t[0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result_t[1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result_t[2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result_t[3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result_t[4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result_t[5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result_t[6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result_t[7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result_t[8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result_t[9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result_t[10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result_t[11]:.3f}"
            )
        if all_eval_result_s is not None:
            logger.info(
                f"val: epoch: {checkpoint_s['epoch']:0>5d}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result_s[0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result_s[1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result_s[2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result_s[3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result_s[4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result_s[5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result_s[6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result_s[7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result_s[8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result_s[9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result_s[10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result_s[11]:.3f}"
            )


        return


    best_map_t = 0.0
    best_map_s = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume_t):
        logger.info(f"start resuming model from {args.resume_t}")
        checkpoint_t = torch.load(args.resume_t, map_location=torch.device('cpu'))
        start_epoch += checkpoint_t['epoch']
        best_map_t = checkpoint_t['best_map']
        model_t.load_state_dict(checkpoint_t['model_state_dict'])
        optimizer_t.load_state_dict(checkpoint_t['optimizer_state_dict'])
        scheduler_t.load_state_dict(checkpoint_t['scheduler_state_dict'])
        logger.info(
            f"finish resuming model_t from {args.resume_t}, epoch {checkpoint_t['epoch']}, best_map: {checkpoint_t['best_map']}, "
            f"loss: {checkpoint_t['loss']:3f}, cls_loss: {checkpoint_t['cls_loss']:2f}, reg_loss: {checkpoint_t['reg_loss']:2f}"
        )

    if os.path.exists(args.resume_s):
        logger.info(f"start resuming model from {args.resume_s}")
        checkpoint_s = torch.load(args.resume_s, map_location=torch.device('cpu'))
        best_map_s = checkpoint_s['best_map']
        model_s.load_state_dict(checkpoint_s['model_state_dict'])
        optimizer_s.load_state_dict(checkpoint_s['optimizer_state_dict'])
        scheduler_s.load_state_dict(checkpoint_s['scheduler_state_dict'])
        logger.info(
            f"finish resuming model_s from {args.resume_s}, epoch {checkpoint_s['epoch']}, best_map: {checkpoint_s['best_map']}, "
            f"loss: {checkpoint_s['loss']:3f}, cls_loss: {checkpoint_s['cls_loss']:2f}, reg_loss: {checkpoint_s['reg_loss']:2f}"
        )

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):


        cls_losses_t, reg_losses_t, losses_t,cls_losses_s, reg_losses_s, losses_s = train(train_loader, model_t, criterion_t,
                                                                                          optimizer_t, scheduler_t,model_s, criterion_s,
                                                                                          optimizer_s, scheduler_s, epoch,
                                                                                          logger, args,emdloss)
        logger.info(
            f"train: epoch {epoch:0>3d}, cls_loss_t: {cls_losses_t:.2f}, reg_loss: {reg_losses_t:.2f}, loss: {losses_t:.2f},"
            f"cls_loss_s: {cls_losses_s:.2f}, reg_loss: {reg_losses_s:.2f}, loss: {losses_s:.2f}"
        )
        
        if epoch % 1 == 0 or epoch == args.epochs:
            logger.info(f"start eval.")
            all_eval_result_t = validate(Config.val_dataset, model_t, decoder_t,args)

            all_eval_result_s = validate(Config.val_dataset, model_s, decoder_s,args)
            logger.info(f"eval done.")
            if all_eval_result_t is not None:
                logger.info(
                    f"val: epoch: {epoch:0>5d}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result_t[0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result_t[1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result_t[2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result_t[3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result_t[4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result_t[5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result_t[6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result_t[7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result_t[8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result_t[9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result_t[10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result_t[11]:.3f},"
                    f"IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result_s[0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result_s[1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result_s[2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result_s[3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result_s[4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result_s[5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result_s[6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result_s[7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result_s[8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result_s[9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result_s[10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result_s[11]:.3f}"
                )
                if all_eval_result_t[0] > best_map_t:
                    torch.save(model_t.module.state_dict(),
                               os.path.join(args.checkpoints, "best_t_MAP_{}_MAR_{}.pth".format(all_eval_result_t[0],all_eval_result_t[8])))
                    best_map_t = all_eval_result_t[0]

                if all_eval_result_s[0] > best_map_s:
                    torch.save(model_s.module.state_dict(),
                               os.path.join(args.checkpoints, "best_s_MAP_{}_MAR_{}.pth".format(all_eval_result_s[0],
                                                                                                all_eval_result_s[8])))
                    best_map_s = all_eval_result_s[0]
            torch.save(
                {
                    'epoch': epoch,
                    'best_map': best_map_t,
                    'cls_loss': cls_losses_t,
                    'reg_loss': reg_losses_t,
                    'loss': losses_t,
                    'model_state_dict': model_t.state_dict(),
                    'optimizer_state_dict': optimizer_t.state_dict(),
                    'scheduler_state_dict': scheduler_t.state_dict(),
                }, os.path.join(args.checkpoints, 'epoch{}_t_MAP_{}_MAR_{}.pth'.format(epoch,all_eval_result_t[0],all_eval_result_t[8])))

            torch.save(
                {
                    'epoch': epoch,
                    'best_map': best_map_s,
                    'cls_loss': cls_losses_s,
                    'reg_loss': reg_losses_s,
                    'loss': losses_s,
                    'model_state_dict': model_s.state_dict(),
                    'optimizer_state_dict': optimizer_s.state_dict(),
                    'scheduler_state_dict': scheduler_s.state_dict(),
                }, os.path.join(args.checkpoints, 'epoch{}_s_MAP_{}_MAR_{}.pth'.format(epoch, all_eval_result_s[0],
                                                                                         all_eval_result_s[8])))

    logger.info(f"finish training, best_map: {best_map_t:.3f}")
    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")



def train(train_loader, model_t, criterion_t, optimizer_t, scheduler_t, model_s, criterion_s, optimizer_s, scheduler_s,epoch, logger,
          args,emdloss):
    cls_losses_t, reg_losses_t, losses_t = [], [], []
    cls_losses_s, reg_losses_s, losses_s = [], [], []
    # switch to train mode
    model_t.train()
    model_s.train()

    iters = len(train_loader.dataset) // args.per_node_batch_size
    prefetcher = COCODataPrefetcher(train_loader)
    images, annotations = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, annotations = images.cuda().float(), annotations.cuda()
        cls_heads_t, reg_heads_t, batch_anchors_t,C6_t = model_t(images)
        cls_loss_t, reg_loss_t = criterion_t(cls_heads_t, reg_heads_t, batch_anchors_t,
                                             annotations)
        

        cls_heads_s, reg_heads_s, batch_anchors_s,C6_s = model_s(images)
        cls_loss_s, reg_loss_s = criterion_s(cls_heads_s, reg_heads_s, batch_anchors_s,
                                             annotations)

        kl_s = kl_loss(cls_heads_t[0], cls_heads_s[0]) + kl_loss(cls_heads_t[1], cls_heads_s[1]) + kl_loss(cls_heads_t[2],
                                                                                                           cls_heads_s[
                                                                                                               2]) + kl_loss(
            cls_heads_t[3], cls_heads_s[3]) + kl_loss(cls_heads_t[4], cls_heads_s[4])

        kl_t = kl_loss(cls_heads_s[0], cls_heads_t[0]) + kl_loss(cls_heads_s[1], cls_heads_t[1]) + kl_loss(
            cls_heads_s[2], cls_heads_t[2]) + kl_loss(
            cls_heads_s[3], cls_heads_t[3]) + kl_loss(cls_heads_s[4], cls_heads_t[4])
        loss_t = cls_loss_t + reg_loss_t+0.1*kl_t


        emd=emdloss(C6_t, C6_s)

        loss_s = cls_loss_s + reg_loss_s+0.1*kl_s+0.01*emd
        if cls_loss_t == 0.0 or reg_loss_t == 0.0:
            optimizer_t.zero_grad()
            continue
        if cls_loss_s == 0.0 or reg_loss_s == 0.0:
            optimizer_s.zero_grad()
            continue

        if args.apex:
            with amp.scale_loss(loss_t, optimizer_t) as scaled_loss_t:
                scaled_loss_t.backward(retain_graph=True)

            with amp.scale_loss(loss_s, optimizer_s) as scaled_loss_s:
                scaled_loss_s.backward()
        else:
            loss_t.backward(retain_graph=True)
            loss_s.backward()

        torch.nn.utils.clip_grad_norm_(model_t.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 0.1)
        optimizer_t.step()
        optimizer_t.zero_grad()

        optimizer_s.step()
        optimizer_s.zero_grad()

        cls_losses_t.append(cls_loss_t.item())
        reg_losses_t.append(reg_loss_t.item())
        losses_t.append(loss_t.item())

        cls_losses_s.append(cls_loss_s.item())
        reg_losses_s.append(reg_loss_s.item())
        losses_s.append(loss_s.item())

        images, annotations = prefetcher.next()

        if iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>5d}, {iters:0>5d}], cls_loss_t: {cls_loss_t.item():.2f}, reg_loss_t: {reg_loss_t.item():.2f}, loss_total_t: {loss_t.item():.2f},"
                f" cls_loss_s: {cls_loss_s.item():.2f}, reg_loss_s: {reg_loss_s.item():.2f}, loss_total_s: {loss_s.item():.2f}"
            )

        iter_index += 1

    scheduler_t.step(np.mean(losses_t))
    scheduler_s.step(np.mean(losses_s))

    return np.mean(cls_losses_t), np.mean(reg_losses_t), np.mean(losses_t), np.mean(cls_losses_s), np.mean(reg_losses_s), np.mean(losses_s)


if __name__ == '__main__':
    main()
