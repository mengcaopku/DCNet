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
from utils.parsing_metrics import *
from utils.utils import *


def save_bbox(bbox, im_id, save_path='./visulizations/'):
    n = bbox.shape[0]
    save_path=save_path + 'pred_bbox'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file_path = os.path.join(save_path, 'pred_bbox.txt')

    for ii in range(n):
        im_path = im_id[ii]
        write_file = open(save_file_path, 'a+')
        bbox =  bbox.numpy()
        print('write...%s'%im_path)
        write_file.write('%s,%d,%d,%d,%d\r\n'%(im_path, int(bbox[ii][0]), int(bbox[ii][1]), int(bbox[ii][2]), int(bbox[ii][3]) ))
        write_file.close()
       #(bbox[ii,0], bbox[ii,1]), (bbox[ii,2], bbox[ii,3])
       



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
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--lstm', dest='lstm', default=False, action='store_true', help='if use lstm as language module instead of bert')
    parser.add_argument('--num_frame_k', default=5, type=int, help='num frames of reference')
    parser.add_argument('--cache_dir', default='./cache', type=str, help='cache file dir path')


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


    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    test_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         testmode=True,
                         split='test',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm,
                         num_frame_k=args.num_frame_k)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

    ## Model
    ## input ifcorpus=None to use bert as text encoder
    ifcorpus = None

    
    if args.test:
        _ = post_processing(test_loader, args.size_average, args.num_frame_k)
        exit(0)
    


def read_data(img_path, frm_idx, batch_idx, center_im=None, center_im_idx=None, cache_dir = './cache'):
    vid_name = img_path.split('/')[-2]
    img_name = img_path.split('/')[-1]

    cache_path = os.path.join(cache_dir, vid_name)

    save_file = os.path.join(cache_path, img_name.split('.JPEG')[0] + '_' + str(batch_idx) + '.pth')
   
    invalid = -1
    if not os.path.exists(save_file):

        save_file = os.path.join(cache_path, center_im.split('/')[-1].split('.JPEG')[0] + '_' + str(center_im_idx) + '.pth')
        invalid = frm_idx

    data = torch.load(save_file)

    pred_bbox_topk = data['pred_bbox_topk'] # tensor: topk x 1 x 4
    pred_score_topk = torch.tensor(data['pred_score_topk'], dtype=torch.float) # list: [.,.,.,]

    visu_feat = data['visu_feat'] # tensor: topk x 1 x 512

    return pred_bbox_topk, pred_score_topk, visu_feat, invalid


def post_processing(val_loader, size_average, topk, mode='test'):
    # try to visualize the top-k bboxes in the grounding results
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()

    
    end = time.time()
    
    for batch_idx, (imgs, word_id, word_mask, bbox, ratio, dw, dh, im_id, phrase) in enumerate(val_loader):
        
        ref_frame = args.num_frame_k
        cache_dir = args.cache_dir

        center_frm_idx = int(ref_frame/2)
        im_name = im_id[center_frm_idx][0]
        pred_bbox_topk, _, visu_feat, _ = read_data(im_name, center_frm_idx, batch_idx, cache_dir=cache_dir)

        
        # read reference frame
        visu_feat_set = []
        pred_score_topk_set = []
        offset_list = list(range(-center_frm_idx, center_frm_idx+1))#[-2,-1,0,1,2]
        invalid = []
        for offset, frm_idx in zip(offset_list, range(ref_frame)):
            # if(frm_idx == center_frm_idx):
                # continue
            im_name_ref = im_id[frm_idx][0]

            pred_bbox_topk_ref, pred_score_topk_ref, visu_feat_ref, invalid_idx = read_data(im_name_ref, frm_idx, batch_idx + offset, center_im=im_name, center_im_idx=batch_idx, cache_dir=cache_dir)
            if(invalid_idx > -1):
                invalid.append(invalid_idx)
            # pred_bbox_topk_ref: tensor: topk x 1 x 4
            # pred_score_topk_ref: list: [.,.,.,]
            # visu_feat_ref: tensor: topk x 1 x 512
            visu_feat_set.append(visu_feat_ref)
            pred_score_topk_set.append(pred_score_topk_ref)


        refer_visu_feat = torch.cat(visu_feat_set, dim=1) # topk x ref_frame x feat_dim
        center_frame_feat = visu_feat.unsqueeze(1) # topk x feat_dim
        refer_pred_score_topk =  torch.stack(pred_score_topk_set).permute(1,0) # topk x ref_frame

        # calculate similarity
        _, _, feat_dim = refer_visu_feat.shape
        refer_visu_feat = refer_visu_feat.view(-1, feat_dim) 
        refer_visu_feat = refer_visu_feat.permute(1, 0) # feat_dim x (topk x ref_frame)
        center_frame_feat = center_frame_feat.view(-1, feat_dim) # topk x feat_dim

        sim_weight = torch.bmm(center_frame_feat.unsqueeze(0), refer_visu_feat.unsqueeze(0)) # topk x (topk x ref_frame)
        # max pool the score to choose the highest score for the reference frame
        sim_weight = sim_weight.reshape(topk, topk, ref_frame) # topk x topk x ref_frame
        sim_weight_maxpool, sim_idx =  sim_weight.max(dim=1)

        refer_score = refer_pred_score_topk.gather(0, torch.LongTensor(sim_idx)) # topk x ref_frame


        sim_weight_maxpool = F.softmax(sim_weight_maxpool,dim=1) # topk x ref_frame

        if len(invalid) > 0:
            sim_weight_maxpool[:,invalid] = 0

        fused_pred_score = torch.sum(sim_weight_maxpool * refer_score, dim=1) # topk
        max_conf = fused_pred_score.max()

        (topk_idx,) = np.where(fused_pred_score.cpu().numpy() == max_conf.cpu().numpy())
        topk_idx = int(topk_idx[0])

        pred_bbox = pred_bbox_topk[topk_idx]
   


        #f.write('%d,%s\r\n'%(batch_idx, phrase[0]))
        imgs = imgs.contiguous().view(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
        imgs = imgs.cuda()

        bbox = bbox[:,center_frm_idx]
        bbox = bbox.cuda()
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        imgs = imgs[center_frm_idx].unsqueeze(0)
        phrase = phrase[-1]
        ratio = ratio[:,-1]
        dw = dw[:,-1]
        dh = dh[:,-1]

    
        target_bbox = bbox.data.cpu()

        target_bbox[:,0], target_bbox[:,2] = (target_bbox[:,0]-dw)/ratio, (target_bbox[:,2]-dw)/ratio
        target_bbox[:,1], target_bbox[:,3] = (target_bbox[:,1]-dh)/ratio, (target_bbox[:,3]-dh)/ratio
       
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

       
        target_bbox[:,:2], target_bbox[:,2], target_bbox[:,3] = \
            torch.clamp(target_bbox[:,:2], min=0), torch.clamp(target_bbox[:,2], max=img_np.shape[3]), torch.clamp(target_bbox[:,3], max=img_np.shape[2])

        iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        # accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/1
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/1

        acc.update(accu, imgs.size(0))
        # acc_center.update(accu_center, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print('sim score:', sim_score)
        # print('loc score:', loc_score)

        if args.save_plot:
            if batch_idx%1==0:
                save_bbox(pred_bbox, im_id, save_path='./visulizations/%s/'%args.savename)
                
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, \
                    acc=acc, miou=miou)
            print(print_str)
            logging.info(print_str)
    
    
    print(acc.avg, miou.avg)
    logging.info("%f,%f"%(acc.avg, float(miou.avg)))
    return acc.avg



if __name__ == "__main__":
    main()
 
