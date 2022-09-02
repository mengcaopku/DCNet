import os
import sys
import argparse
import shutil
import time
import random
import gc
import json
from distutils.version import LooseVersion
import scipy.misc
import logging

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import ResizeImage, ResizeAnnotation

# use normalized location embedding in this script
from dataset.vid_loader import *
from model.test_DCNet_model import *
from utils.parsing_metrics import *
from utils.utils import *


def save_grounding_results(bbox, target_bbox, input, phrase, mode, batch_start_index, \
    merge_pred=None, pred_conf_visu=None, save_path='./visulizations/'):
    n = input.shape[0]
    save_path=save_path+mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    input=input.data.cpu().numpy()
    input=input.transpose(0,2,3,1)
    for ii in range(n):
        #os.system('mkdir -p %s/sample_%d'%(save_path,batch_start_index+ii))
        imgs = input[ii,:,:,:].copy()
        imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        # imgs = imgs.transpose(2,0,1)
        imgs = np.array(imgs, dtype=np.float32)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        cv2.rectangle(imgs, (bbox[ii,0], bbox[ii,1]), (bbox[ii,2], bbox[ii,3]), (255,0,0), 2)
        cv2.rectangle(imgs, (target_bbox[ii,0], target_bbox[ii,1]), (target_bbox[ii,2], target_bbox[ii,3]), (0,255,0), 2)
        cv2.putText(imgs, phrase[0], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,250), 1)
        print('saving....%s/sample_%d.jpg\n'%(save_path,batch_start_index+ii))
        cv2.imwrite('%s/sample_%d.jpg'%(save_path,batch_start_index+ii),imgs)
    #cv2.imwrite('%s/sample_%d/pred_yolo.png'%(save_path,batch_start_index+ii),imgs)



def build_target(raw_coord, pred):
    coord_list, bbox_list, bbox_center_list = [],[],[]
    for scale_ii in range(len(pred)):
        coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
        batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii))
        coord[:,0] = (raw_coord[:,0] + raw_coord[:,2])/(2*args.size)
        coord[:,1] = (raw_coord[:,1] + raw_coord[:,3])/(2*args.size)
        coord[:,2] = (raw_coord[:,2] - raw_coord[:,0])/(args.size)
        coord[:,3] = (raw_coord[:,3] - raw_coord[:,1])/(args.size)
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0),3,5,grid, grid))
        bbox_center_list.append(torch.zeros(coord.size(0),5,grid,grid))

    best_n_list, best_gi, best_gj = [],[],[]
    #import IPython
    #IPython.embed()
    for ii in range(batch):
        anch_ious = []
        for scale_ii in range(len(pred)):
            batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii))
            gi = coord_list[scale_ii][ii,0].long()
            gj = coord_list[scale_ii][ii,1].long()
            tx = coord_list[scale_ii][ii,0] - gi.float()
            ty = coord_list[scale_ii][ii,1] - gj.float()

            gw = coord_list[scale_ii][ii,2]
            gh = coord_list[scale_ii][ii,3]

            anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            ## Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n//3

        batch, grid = raw_coord.size(0), args.size//(32/(2**best_scale))
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]

        gi = coord_list[best_scale][ii,0].long()
        gj = coord_list[best_scale][ii,1].long()
        tx = coord_list[best_scale][ii,0] - gi.float()
        ty = coord_list[best_scale][ii,1] - gj.float()
        gw = coord_list[best_scale][ii,2]
        gh = coord_list[best_scale][ii,3]
        tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        bbox_center_list[best_scale][ii,:, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = Variable(bbox_list[ii].cuda())
        bbox_center_list[ii] = Variable(bbox_center_list[ii].cuda())
    return bbox_list, best_gi, best_gj, best_n_list, bbox_center_list

def main():
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=16, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--size_average', dest='size_average', 
                        default=False, action='store_true', help='size_average')
    parser.add_argument('--size', default=256, type=int, help='image size')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: sgd, adam, RMSprop')
    parser.add_argument('--print_freq', '-p', default=2000, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--save_plot', dest='save_plot', default=False, action='store_true', help='save visulization plots')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--cache', dest='cache', default=False, action='store_true', help='save cache files')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--lstm', dest='lstm', default=False, action='store_true', help='if use lstm as language module instead of bert')
    parser.add_argument('--num_frame_k', default=5, type=int, help='num frames of reference')


    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
    ## following anchor sizes calculated by kmeans under args.anchor_imsize=416
    if args.dataset=='refeit':
        anchors = '30,36,  78,46,  48,86,  149,79,  82,148,  331,93,  156,207,  381,163,  329,285'
    elif args.dataset=='flickr':
        anchors = '29,26,  55,58,  137,71,  82,121,  124,205,  204,132,  209,263,  369,169,  352,294'
    else:
        anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

    ## save logs
    if args.savename=='default':
        args.savename = 'model_%s_batch%d'%(args.dataset,args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    if args.test:
        logging.basicConfig(level=logging.DEBUG, filename="./logs/%s_test"%args.savename, filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
    else:
        logging.basicConfig(level=logging.DEBUG, filename="./logs/%s"%args.savename, filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='train',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm,
                         augment=True)
    val_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='test',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    num_frame_k = args.num_frame_k

    test_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         testmode=True,
                         split='test',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm,
                         num_frame_k=num_frame_k)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

    ## Model
    ## input ifcorpus=None to use bert as text encoder
    ifcorpus = None
    if args.lstm:
        ifcorpus = train_dataset.corpus
    model = grounding_model(corpus=ifcorpus, light=args.light, emb_size=args.emb_size, coordmap=True,\
        bert_model=args.bert_model, dataset=args.dataset)
    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            assert (len([k for k, v in pretrained_dict.items()])!=0)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("=> loaded pretrain model at {}"
                  .format(args.pretrain))
            logging.info("=> loaded pretrain model at {}"
                  .format(args.pretrain))
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss)))
            logging.info("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))

    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    visu_param = model.module.visumodel.parameters()
    rest_param = [param for param in model.parameters() if param not in visu_param]
    visu_param = list(model.module.visumodel.parameters())
    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
    sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
    print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    ## optimizer; rmsprop default
    if args.optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99)
    else:
        optimizer = torch.optim.RMSprop([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.}], lr=args.lr, weight_decay=0.0005)

    ## training and testing
    best_accu = -float('Inf')
    if args.test:
        _ = test_epoch(test_loader, model, args.size_average)
        #exit(0)
    
    if args.cache:
        _ =  save_cache(test_loader, model, args.size_average, num_frame_k)
        #exit(0)

    exit(0)


def test_epoch(val_loader, model, size_average, mode='test'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()
    mid_idx = int(args.num_frame_k / 2)
    #f = open('test_phrase.txt', 'a+')
    for batch_idx, (imgs, word_id, word_mask, bbox, ratio, dw, dh, im_id, phrase) in enumerate(val_loader):
        
        #f.write('%d,%s\r\n'%(batch_idx, phrase[0]))
        imgs = imgs.contiguous().view(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])

        word_id = word_id[:,mid_idx]
        word_mask = word_mask[:,mid_idx]
        bbox = bbox[:,mid_idx]


        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        with torch.no_grad():
            ## Note LSTM does not use word_mask
            pred_anchor, sim_score, loc_score, fvisu, obj_score = model(image, word_id, word_mask, args.num_frame_k)


        imgs = imgs[mid_idx].unsqueeze(0)
        phrase = phrase[-1]
        ratio = ratio[:,-1]
        dw = dw[:,-1]
        dh = dh[:,-1]


        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii][-1].unsqueeze(0)
            sim_score[ii] = sim_score[ii][-1].unsqueeze(0)
            loc_score[ii] = loc_score[ii][-1].unsqueeze(0)
            obj_score[ii] = obj_score[ii][-1].unsqueeze(0)

        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj, best_n_list, dummy = build_target(bbox, pred_anchor)
        
        ## test: convert center+offset to box prediction
        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(1,-1))
            gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(1,-1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(1,4)

        pred_gi, pred_gj, pred_best_n = [],[],[]
        for ii in range(1):
            if max_loc[ii] < 3*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(1,3,grid,grid).data.cpu().numpy()
            max_conf_ii = max_conf.data.cpu().numpy()

            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size


        pred_bbox = xywh2xyxy(pred_bbox)
        target_bbox = bbox.data.cpu()
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio
        target_bbox[:,0], target_bbox[:,2] = (target_bbox[:,0]-dw)/ratio, (target_bbox[:,2]-dw)/ratio
        target_bbox[:,1], target_bbox[:,3] = (target_bbox[:,1]-dh)/ratio, (target_bbox[:,3]-dh)/ratio
        #print(target_bbox)
        ## convert pred, gt box to original scale with meta-info
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)
        #print(img_np.shape)
        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        ## also revert image for visualization
        #print(new_shape)
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))

        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
        target_bbox[:,:2], target_bbox[:,2], target_bbox[:,3] = \
            torch.clamp(target_bbox[:,:2], min=0), torch.clamp(target_bbox[:,2], max=img_np.shape[3]), torch.clamp(target_bbox[:,3], max=img_np.shape[2])

        iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/1
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/1

        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print('sim score:', sim_score)
        # print('loc score:', loc_score)

        if args.save_plot:
            if batch_idx%1==0:        
                save_grounding_results(pred_bbox,target_bbox,img_np, phrase, 'test', batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.savename)

        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, \
                    acc=acc, acc_c=acc_center, miou=miou)
            print(print_str)
            logging.info(print_str)
    print(best_n_list, pred_best_n)
    print(np.array(target_gi), np.array(pred_gi))
    print(np.array(target_gj), np.array(pred_gj),'-')
    print(acc.avg, miou.avg,acc_center.avg)
    logging.info("%f,%f,%f"%(acc.avg, float(miou.avg),acc_center.avg))
    return acc.avg

def get_pred_bbox(pred_conf_list, max_conf, max_loc, pred_anchor, ratio, dw, dh, img_np):

    pred_bbox = torch.zeros(1,4)

    pred_gi, pred_gj, pred_best_n = [],[],[]
    for ii in range(1):
        if max_loc[ii] < 3*(args.size//32)**2:
            best_scale = 0
        elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
            best_scale = 1
        else:
            best_scale = 2

        grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]

        pred_conf = pred_conf_list[best_scale].view(1,3,grid,grid).data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()

        # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
        (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
        best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
        pred_gi.append(gi)
        pred_gj.append(gj)
        pred_best_n.append(best_n+best_scale*3)

        pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
        pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
        pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
        pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size

        pred_bbox = xywh2xyxy(pred_bbox)
       
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio
        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
    
    return pred_bbox


def save_cache(val_loader, model, size_average, topk, mode='test'):
    # try to visualize the top-k bboxes in the grounding results
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()
    
    for batch_idx, (imgs, word_id, word_mask, bbox, ratio, dw, dh, im_id, phrase) in enumerate(val_loader):

        imgs = imgs.contiguous().view(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])

        word_id = word_id[:,2]
        word_mask = word_mask[:,2]
        bbox = bbox[:,2]

        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        with torch.no_grad():
            ## Note LSTM does not use word_mask
            pred_anchor, sim_score, loc_score, fvisu, only_obj = model(image, word_id, word_mask, topk)


        imgs = imgs[2].unsqueeze(0)
        phrase = phrase[-1]
        ratio = ratio[:,-1]
        dw = dw[:,-1]
        dh = dh[:,-1]
        
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj, best_n_list, dummy = build_target(bbox, pred_anchor)
        
        ## test: convert center+offset to box prediction
        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(1,-1))
            gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(1,-1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        # max_conf, max_loc = torch.max(pred_conf, dim=1)
        # find the top-k conf and loc
        max_conf_topk, max_loc_topk = torch.topk(pred_conf, k=topk, dim=1)
        # max_conf_topk.shape: 1 x topk
        # max_loc_topk.shape: 1 x topk
        pred_bbox_topk = []
        pred_score_topk = []
        best_scale_topk = []
        best_n_topk = []
        gj_topk = []
        gi_topk = []
        visu_feat = []

        target_bbox = bbox.data.cpu()
        target_bbox[:,0], target_bbox[:,2] = (target_bbox[:,0]-dw)/ratio, (target_bbox[:,2]-dw)/ratio
        target_bbox[:,1], target_bbox[:,3] = (target_bbox[:,1]-dh)/ratio, (target_bbox[:,3]-dh)/ratio
        #print(target_bbox)
        ## convert pred, gt box to original scale with meta-info
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)
        #print(img_np.shape)
        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        ## also revert image for visualization
        #print(new_shape)
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))


        for ii in range(topk):
            pred_bbox, pred_score_, best_scale_, best_n_,  gj_, gi_ = get_topk_pred_bbox(pred_conf_list, sim_score, loc_score, max_conf_topk[:,ii], max_loc_topk[:,ii], pred_anchor, ratio, dw, dh, img_np)
            pred_bbox_topk.append(pred_bbox)
            pred_score_topk.append(pred_score_)
            best_scale_topk.append(best_scale_)
            gj_topk.append(gj_)
            gi_topk.append(gi_)
            visu_feat.append(fvisu[best_scale_][:,:,gj_, gi_].cpu())

        img_path = im_id[2][0]
        vid_name = img_path.split('/')[-2]
        img_name = img_path.split('/')[-1]

        cache_path = os.path.join('./cache', args.savename, vid_name)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        #import IPython
        #IPython.embed()
        save_file = os.path.join(cache_path, img_name.split('.JPEG')[0] + '_' + str(batch_idx) + '.pth')
        save_item = {}
        save_item['pred_bbox_topk'] = torch.stack(pred_bbox_topk)
        save_item['pred_score_topk'] = pred_score_topk
        save_item['visu_feat'] = torch.stack(visu_feat)
        print(save_file)
        torch.save(save_item, save_file)


def get_topk_pred_bbox(pred_conf_list, sim_score, loc_score, max_conf, max_loc, pred_anchor, ratio, dw, dh, img_np):

    pred_bbox = torch.zeros(1,4)

    pred_gi, pred_gj, pred_best_n = [],[],[]
    for ii in range(1):
        if max_loc[ii] < 3*(args.size//32)**2:
            best_scale = 0
        elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
            best_scale = 1
        else:
            best_scale = 2

        grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]

        pred_conf = pred_conf_list[best_scale].view(1,3,grid,grid).data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()

        # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
        (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
        best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
        pred_gi.append(gi)
        pred_gj.append(gj)
        pred_best_n.append(best_n+best_scale*3)

        pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
        pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
        pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
        pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size

        pred_bbox = xywh2xyxy(pred_bbox)
       
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio
        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
        pred_score_ = max_conf_ii[ii]

    
    return pred_bbox, pred_score_, best_scale, best_n, gj, gi



if __name__ == "__main__":
    main()
 
