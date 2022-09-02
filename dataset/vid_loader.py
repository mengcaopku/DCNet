import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
# import h5py
import numpy as np
import numpy
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import utils
from utils import Corpus

import argparse
import collections
import logging
import json
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from utils.transforms import letterbox, random_affine


sys.modules['utils'] = utils

cv2.setNumThreads(0)

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

def bbox_randscale(bbox, miniou=0.75):
    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    scale_shrink = (1-math.sqrt(miniou))/2.
    scale_expand = (math.sqrt(1./miniou)-1)/2.
    w1,h1 = random.uniform(-scale_expand, scale_shrink)*w, random.uniform(-scale_expand, scale_shrink)*h
    w2,h2 = random.uniform(-scale_shrink, scale_expand)*w, random.uniform(-scale_shrink, scale_expand)*h
    bbox[0],bbox[2] = bbox[0]+w1,bbox[2]+w2
    bbox[1],bbox[3] = bbox[1]+h1,bbox[3]+h2
    return bbox

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def getChunk(img_path, split,num_frame_k=2):
    # img_path is actually a video-path
    vid_list = torch.load(img_path)

    images = list()
    num_floor = int(numpy.floor(num_frame_k/2))
    num_ceil = int(numpy.ceil(num_frame_k/2))
    for vids in vid_list:
        if split=="train":
            num=random.randint(0,len(vids)-1)
            vid=vids[num]
        else:
            vid=vids

        vid_len=len(vid)
        
        for img_idx in range(vid_len):
            chunk = list()
            imgs_path = list()
            bboxs = list()
            annotations = list()
            if img_idx - num_floor<0 :
                continue
            if img_idx + num_ceil>vid_len-1:
                continue
            for i in range(img_idx - num_floor, img_idx + num_ceil):
                img_path=vid[numpy.clip(i,0,vid_len-1)][0]
                imgs_path.append(img_path)
                bboxs.append(vid[numpy.clip(i,0,vid_len-1)][1])
                annotations.append(vid[numpy.clip(i,0,vid_len-1)][2])
             
            chunk.append(imgs_path)
            chunk.append(bboxs) # bbox
            chunk.append(annotations) # annotation
            images.append(tuple(chunk))


    return images


class DatasetNotFoundError(Exception):
    pass

class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')}, 
        'VID': {'splits':('train', 'test')}, # add the split definition for VID dataset
        'VID_noun': {'splits':('train', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            # 'splits': ('train', 'val'),
            # 'params': {'dataset': 'refcocog', 'split_by': 'google'}
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit', imsize=256,
                 transform=None, augment=False, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, bert_model='bert-base-uncased',num_frame_k=2):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.lstm = lstm
        self.corpus = Corpus()
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.augment=augment
        self.return_idx=return_idx
        self.num_frame_k = num_frame_k

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        elif self.dataset == 'VID': # define the data path for VID dataset
            self.dataset_root = osp.join(self.data_root, 'VID')
            self.im_dir = osp.join(self.dataset_root, 'VID')
        elif self.dataset == 'VID_noun': # define the data path for VID_noun dataset
            self.dataset_root = osp.join(self.data_root, 'VID_noun')
            self.im_dir = osp.join(self.dataset_root, 'VID_noun')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
            self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            #self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        corpus_path = osp.join(dataset_path, 'corpus.pth')
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))
        self.corpus = torch.load(corpus_path)

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            # Need to change the file path 
            

            imgset_path = './data/VID/VID_video_level_' + split + '.pth'
            
         

      
            self.images += getChunk(imgset_path,split, num_frame_k=self.num_frame_k)


    def exists_dataset(self):
        print(osp.join(self.split_root, self.dataset))
        return osp.exists(osp.join(self.split_root, self.dataset))

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def pull_item(self, idx):
        if self.dataset == 'flickr' or self.dataset == 'VID' or self.dataset == 'VID_noun' or self.dataset == 'gref':
            #print(self.images[idx])
            img_files, bbox_list, phrase_list = self.images[idx]
        else:
            img_files, _, bbox_list, phrase_list, attri_list = self.images[idx]


        #-----------------------------------------------------------------------
        #print("img_files",img_files)
        #-----------------------------------------------------------------------


        bboxs = []
        ## box format: to x1y1x2y2
        for bbox in bbox_list:
            if not (self.dataset == 'referit' or self.dataset == 'flickr' or self.dataset == 'VID' or self.dataset == 'VID_noun'):
                bbox = np.array(bbox, dtype=int)
                bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
            else:
                bbox = np.array(bbox, dtype=int)
            bboxs.append(bbox)

        imgs = []
        for img_file in img_files:
            if self.dataset == 'VID' or self.dataset == 'VID_noun':
                self.im_dir = ''
                img_path = osp.join(self.im_dir, img_file)
            else:
                img_path = osp.join(self.im_dir, img_file)
            img = cv2.imread(img_path)
            # print(img_path)
            ## duplicate channel if gray image
            if img.shape[-1] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.stack([img] * 3)
            imgs.append(img)
        return imgs, phrase_list, bboxs, img_files

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        imgs, phrases, bboxs, img_files = self.pull_item(idx)
        ori_phrases = list()
        for i in range(self.num_frame_k):
            phrases[i] = phrases[i].lower()
            ori_phrases.append(phrases[i].lower())
        if self.augment:
            augment_flip, augment_hsv, augment_affine = True,True,True


        ratios = []
        dws = []
        dhs = []
        ## seems a bug in torch transformation resize, so separate in advance
        if self.augment:
            ## random horizontal flip
            h,w = imgs[0].shape[0], imgs[0].shape[1]

            if augment_flip and random.random() > 0.5:
                for i in range(self.num_frame_k):
                    imgs[i] = cv2.flip(imgs[i], 1)
                    bboxs[i][0], bboxs[i][2] = w-bboxs[i][2]-1, w-bboxs[i][0]-1
                    phrases[i] = phrases[i].replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.50
                for i in range(self.num_frame_k):
                    img_hsv = cv2.cvtColor(cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                    S = img_hsv[:, :, 1].astype(np.float32)
                    V = img_hsv[:, :, 2].astype(np.float32)
                    a = (random.random() * 2 - 1) * fraction + 1
                    if a > 1:
                        np.clip(S, a_min=0, a_max=255, out=S)
                    a = (random.random() * 2 - 1) * fraction + 1
                    V *= a
                    if a > 1:
                        np.clip(V, a_min=0, a_max=255, out=V)

                    img_hsv[:, :, 1] = S.astype(np.uint8)
                    img_hsv[:, :, 2] = V.astype(np.uint8)
                    img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
                    imgs[i], _, ratio, dw, dh = letterbox(img, None, self.imsize)
                    ratios.append(ratio)
                    dws.append(dw)
                    dhs.append(dh)
                    bboxs[i][0], bboxs[i][2] = bboxs[i][0]*ratio+dw, bboxs[i][2]*ratio+dw
                    bboxs[i][1], bboxs[i][3] = bboxs[i][1]*ratio+dh, bboxs[i][3]*ratio+dh
            ## random affine transformation
            if augment_affine:
                for i in range(self.num_frame_k):
                    imgs[i], _, bboxs[i], M = random_affine(imgs[i], None, bboxs[i], \
                        degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
        else:   ## should be inference, or specified training
            for i in range(self.num_frame_k):
                h,w = imgs[i].shape[0], imgs[i].shape[1]
                imgs[i], _, ratio, dw, dh = letterbox(imgs[i], None, self.imsize)
                bboxs[i][0], bboxs[i][2] = bboxs[i][0]*ratio+dw, bboxs[i][2]*ratio+dw
                bboxs[i][1], bboxs[i][3] = bboxs[i][1]*ratio+dh, bboxs[i][3]*ratio+dh
                ratios.append(ratio)
                dws.append(dw)
                dhs.append(dh)

        ## Norm, to tensor
        if self.transform is not None:
            for i in range(self.num_frame_k):
                imgs[i] = self.transform(imgs[i])
               

        if self.lstm:
            word_id_list = list()
            word_mask_list = list()
            for i in range(self.num_frame_k):
                phrases[i] = self.tokenize_phrase(phrases[i])
                word_id = numpy.array(phrases[i])
                word_mask = np.zeros(word_id.shape)
                word_id_list.append(word_id)
                word_mask_list.append(word_mask)

        else:
            ## encode phrase to bert input
            word_id_list = list()
            word_mask_list = list()
            for i in range(self.num_frame_k):
                examples = read_examples(phrases[i], idx)
                features = convert_examples_to_features(
                    examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
                word_id = features[0].input_ids
                word_mask = features[0].input_mask
                word_id_list.append(word_id)
                word_mask_list.append(word_mask)


        imgs = torch.stack(imgs)
        word_id = numpy.array(word_id_list,dtype=int)
        word_mask = numpy.array(word_mask_list,dtype=int)
        bbox = numpy.array(bboxs,dtype=np.float32)
        ratio = numpy.array(ratios,dtype=np.float32)
        dw = numpy.array(dws,dtype=np.float32)
        dh = numpy.array(dhs,dtype=np.float32)


        if self.testmode:
            return imgs, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0], ori_phrases
        else:
            return imgs, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
            np.array(bbox, dtype=np.float32), ori_phrases

if __name__ == '__main__':
    #import nltk
    import argparse
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    # from utils.transforms import ResizeImage, ResizeAnnotation
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--size', default=416, type=int,
                        help='image size')
    parser.add_argument('--data', type=str, default='./data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--dataset', default='VID', type=str,
                        help='referit/flickr/unc/unc+/gref/VID')
    parser.add_argument('--split', default='train', type=str,
                        help='name of the dataset split used to train')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    args = parser.parse_args()

    torch.manual_seed(13)
    np.random.seed(13)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    refer_val = ReferDataset(data_root=args.data,
                         dataset=args.dataset,
                         split='test',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         testmode=True)
    val_loader = DataLoader(refer_val, batch_size=8, shuffle=False,
                              pin_memory=False, num_workers=0)


    bbox_list=[]
    # for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
    for batch_idx, (imgs, word_id, word_mask, bbox, ratio, dw, dh, idx, phrase) in enumerate(val_loader):
        print(batch_idx)

        # bboxes = (bbox[:,2:]-bbox[:,:2]).numpy().tolist()
        # for bbox in bboxes:
        #     bbox_list.append(bbox)
        # if batch_idx%10000==0 and batch_idx!=0:
        #     print(batch_idx)
