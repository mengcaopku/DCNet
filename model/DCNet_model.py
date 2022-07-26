from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .darknet import *
import torch.nn.functional as F
import argparse
import collections
import logging
import json
import re
import time
## can be commented if only use LSTM encoder
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
# normalize the localization embedding

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

def Crossmodal_corrspondence(lag_feature,vit_feature,lag_vit_map,top_k=1):
    
    lag_vit_map=lag_vit_map.permute(0,2,1)
    vit_feature=vit_feature.permute(0,2,1)
    lag_feature=lag_feature.permute(0,2,1)
    batch=vit_feature.shape[0]
    rows=vit_feature.shape[1]
    values,cols=lag_vit_map.topk(top_k,dim=2,largest=True, sorted=True)

    vit_corrspondence=[]
    lag_corrspondence=[]
    neg_cross=[]
    neg_n=5


    vit_corrspondence_feature=[]
    lag_corrspondence_feature=[]
    neg_cross_feature=[]

   

    for ii in range(batch):
        
        for jj in range(rows):
            #q_list[jj].append(vit_feature[ii][jj,:])
            while jj>len(vit_corrspondence)-1 :
                vit_corrspondence.append([])
                lag_corrspondence.append([])
                neg_cross.append([])
            vit_corrspondence[jj].append(vit_feature[ii][jj,:])

            temp_kk=[]
            for kk in range(top_k):
                temp_kk.append(lag_feature[ii][:,cols[ii][jj][kk]])
            temp_k=torch.stack(temp_kk)

            lag_corrspondence[jj].append(temp_k)



            for index in range(batch):
                random_list=list(range(0, rows))
                if index==ii:
                    random_list.remove(jj)


                neg_feature_index=random.sample(random_list,neg_n)
                temp_nn=[]
                for kk in neg_feature_index:
                    temp_nn.append(vit_feature[index][kk,:])



            temp_neg=torch.stack(temp_nn)

            neg_cross[jj].append(temp_neg)





    
    for ii in range(rows):
        vit_corrspondence_feature.append(torch.stack(vit_corrspondence[ii]))
        lag_corrspondence_feature.append(torch.stack(lag_corrspondence[ii]))
        neg_cross_feature.append(torch.stack(neg_cross[ii]))   


    
        
        
    return   vit_corrspondence_feature,lag_corrspondence_feature,neg_cross_feature






    




class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
               input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size), 
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - inp    iut_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels!=0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)            # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)
        # forward rnn
        output, hidden = self.rnn(embedded)
        # recover
        if self.variable_lengths:
            # recover embedded
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # (batch, max_len, hidden)
            output = output[recover_ixs]

        sent_output = []
        for ii in range(output.shape[0]):
            sent_output.append(output[ii,int(input_lengths_list[ii]-1),:])
        return torch.stack(sent_output, dim=0), output, embedded

class PhraseAttention(nn.Module):
  def __init__(self, input_dim):
    super(PhraseAttention, self).__init__()
    # initialize pivot
    self.fc = nn.Linear(input_dim, 1)

  def forward(self, context, embedded, input_labels):
    """
    Inputs:
    - context : Variable float (batch, seq_len, input_dim)
    - embedded: Variable float (batch, seq_len, word_vec_size)
    - input_labels: Variable long (batch, seq_len)
    Outputs:
    - attn    : Variable float (batch, seq_len)
    - weighted_emb: Variable float (batch, word_vec_size)
    """
    cxt_scores = self.fc(context).squeeze(2) # (batch, seq_len)
    attn = F.softmax(cxt_scores)  # (batch, seq_len), attn.sum(1) = 1.

    # mask zeros
    is_not_zero = (input_labels!=0).float() # (batch, seq_len)
    attn = attn * is_not_zero # (batch, seq_len)
    attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1)) # (batch, seq_len)

    # compute weighted embedding
    attn3 = attn.unsqueeze(1)     # (batch, 1, seq_len)
    weighted_emb = torch.bmm(attn3, embedded) # (batch, 1, word_vec_size)
    weighted_emb = weighted_emb.squeeze(1)    # (batch, word_vec_size)

    return attn, weighted_emb

class grounding_model(nn.Module):
    def __init__(self, corpus=None, emb_size=256, jemb_drop_out=0.1, bert_model='bert-base-uncased', \
     coordmap=True, leaky=False, dataset=None, light=False):
        super(grounding_model, self).__init__()
        self.coordmap = coordmap
        self.light = light
        self.lstm = (corpus is not None)
        self.emb_size = emb_size
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        ## Text model

       
        if self.lstm:
            self.textdim, self.embdim=1024, 512
            self.textmodel = RNNEncoder(vocab_size=len(corpus),
                                          word_embedding_size=self.embdim,
                                          word_vec_size=self.textdim//2,
                                          hidden_size=self.textdim//2,
                                          bidirectional=True,
                                          input_dropout_p=0.2,
                                          variable_lengths=True)
        else:
            self.textmodel = BertModel.from_pretrained(bert_model)

        self.temperature = 10.
        

        # Subjective attention model
        self.sub_attn = PhraseAttention(self.textdim)

        # Location attention model
        self.loc_embedding = torch.nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.loc_text_embedding = torch.nn.Sequential(nn.Linear(1344, self.embdim), nn.BatchNorm1d(self.embdim), nn.ReLU())
        self.loc_attn = PhraseAttention(self.textdim)
       
        ## Mapping module
        self.mapping_visu = nn.Sequential(OrderedDict([
            ('0', ConvBatchNormReLU(1024, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('1', ConvBatchNormReLU(512, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('2', ConvBatchNormReLU(256, emb_size, 1, 1, 0, 1, leaky=leaky))
        ]))
        self.mapping_lang = torch.nn.Sequential(
          nn.Linear(self.textdim, emb_size),
          nn.BatchNorm1d(emb_size),
          nn.ReLU(),
          nn.Dropout(jemb_drop_out),
          nn.Linear(emb_size, emb_size),
          nn.BatchNorm1d(emb_size),
          nn.ReLU(),
        )
        self.corr_conv = nn.Sequential(OrderedDict([
            ('0', torch.nn.Sequential(
                ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ('1', torch.nn.Sequential(
                ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ('2', torch.nn.Sequential(
                ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ]))

        
        self.feature_map=torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=20, out_channels=20, stride=1, kernel_size=3, padding=1, bias=True),
            nn.Softmax(dim=1)
        )
        

        embin_size = emb_size*2
        if self.coordmap:
            embin_size+=8
        if self.light:
            self.fcn_emb = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ]))
            self.fcn_out = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
            ]))
        else:
            self.fcn_emb = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ]))
            self.fcn_out = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
            ]))

    def forward(self, image, word_id, word_mask):
        ## Visual Module
        ## [1024, 13, 13], [512, 26, 26], [256, 52, 52]
        batch_size = image.size(0)
        raw_fvisu = self.visumodel(image)
        #-------------------------------------
        '''
        ipdb> type(raw_fvisu)
        <class 'list'>
        ipdb> len(raw_fvisu)
        3
        ipdb> raw_fvisu[0].shape
        torch.Size([16, 1024, 8, 8])
        '''
        #--------------------------------------
        
        fvisu = []
        for ii in range(len(raw_fvisu)):
            fvisu.append(self.mapping_visu._modules[str(ii)](raw_fvisu[ii]))
            fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)
        
        # calculate image correlation
        # fvisu[0]: batchsize x c1 x h1 x w1 -> batchsize / 2 x 2 x c1 x h1 x w1
        # fvisu[1]: batchsize x c2 x h2 x w2 -> batchsize / 2 x 2 x c2 x h2 x w2
        # fvisu[2]: batchsize x c3 x h3 x w3 -> batchsize / 2 x 2 x c3 x h3 x w3
        fvisu[0] = fvisu[0].contiguous().view(int(batch_size / 2), 2, fvisu[0].shape[-3], fvisu[0].shape[-2], fvisu[0].shape[-1])
        fvisu[1] = fvisu[1].contiguous().view(int(batch_size / 2), 2, fvisu[1].shape[-3], fvisu[1].shape[-2], fvisu[1].shape[-1])
        fvisu[2] = fvisu[2].contiguous().view(int(batch_size / 2), 2, fvisu[2].shape[-3], fvisu[2].shape[-2], fvisu[2].shape[-1])
        
        # split the input
        input1 = []
        input2 = []
        for ii in range(len(raw_fvisu)):
            input1.append(fvisu[ii][:,0])
            input2.append(fvisu[ii][:,1])


        
        #V1
        #----------------Corrspendence Patch------------------------------------------------------------

        patch=input1[0].shape[2]
        feature_frame1=input1[0]
        feature_frame2=input2[0]
        frame_feature_list=[]
        corrspendence_feature_list=[]
        neg_feature_list=[]
        batch=feature_frame1.shape[0]
        feature_patches1=feature_frame1.flatten(-2)
        feature_patches2=feature_frame2.flatten(-2)
        corrspendence_map=torch.bmm(feature_patches1.permute(0,2,1), feature_patches2).flatten(-2)
        top_k=30
        neg_n=10
        
        for ii in range(batch):
            values,index=corrspendence_map[ii].topk(top_k,dim=0,largest=True, sorted=True)

            for jj in range(top_k):
                while jj>len(frame_feature_list)-1:
                    frame_feature_list.append([])
                    corrspendence_feature_list.append([])
                    neg_feature_list.append([])




               
                frame_feature_list[jj].append(feature_patches1[ii][:,index[jj]//(patch*patch)])

                corrspendence_feature_list[jj].append(feature_patches2[ii][:,index[jj]%(patch*patch)])
                
                random_list=list(range(0, patch*patch))
                random_list.remove(index[jj]%(patch*patch))
                neg_feature_index=random.sample(random_list,neg_n)
                temp_nn=[]

                for kk in neg_feature_index:
                    temp_nn.append(feature_patches2[ii][:,kk])

                temp_neg=torch.stack(temp_nn)
                neg_feature_list[jj].append(temp_neg)


        
        frame_feature=[]
        corrspendence_feature=[]
        neg_feature=[]
        for ii in range(top_k):
            frame_feature.append(torch.stack(frame_feature_list[ii]))
            corrspendence_feature.append(torch.stack(corrspendence_feature_list[ii]))
            neg_feature.append(torch.stack(neg_feature_list[ii]))   
        #------------------------------------------------------------------------------------------------
        

        
        fvisu[0] = fvisu[0].contiguous().view(-1, fvisu[0].shape[-3], fvisu[0].shape[-2], fvisu[0].shape[-1])
        fvisu[1] = fvisu[1].contiguous().view(-1, fvisu[1].shape[-3], fvisu[1].shape[-2], fvisu[1].shape[-1])
        fvisu[2] = fvisu[2].contiguous().view(-1, fvisu[2].shape[-3], fvisu[2].shape[-2], fvisu[2].shape[-1])
        

        # co-attention calculation
        key = []
        value = []
        co_attn_1 = []
        co_attn_2 = []
        input1_attn = []
        input2_attn = []
        corr_feat = []

        for ii in range(len(raw_fvisu)):
            b,c,h,w = input1[ii].shape
            key.append(torch.transpose(input1[ii].view(-1, c, h*w), 1, 2).contiguous()) # b x (h*w) x c
            value.append(input2[ii].view(-1, c, h*w).contiguous()) # b x c x (h*w)
            co_attn_1.append(torch.bmm(key[ii],value[ii])) # b x (h*w) x (h*w)
            
            co_attn_2.append(F.softmax(torch.transpose(co_attn_1[ii],1,2) * self.temperature, dim=1))
            co_attn_1[ii] = F.softmax(co_attn_1[ii] * self.temperature, dim=1) # b x (h*w) x (h*w)

            input1_attn.append(torch.bmm(value[ii],co_attn_2[ii]).view(-1, c, h, w).contiguous())  # b x c x h x w
            input2_attn.append(torch.bmm(torch.transpose(key[ii],1,2), co_attn_1[ii]).view(-1, c, h, w).contiguous()) # b x c x h x w

            input1[ii] = torch.cat([input1[ii], input1_attn[ii]],1).unsqueeze(1) # b x 1 x (2c) x h x w
            input2[ii] = torch.cat([input2[ii], input2_attn[ii]],1).unsqueeze(1) # b x 1 x (2c) x h x w
            
            corr_feat.append(torch.cat([input1[ii],input2[ii]],dim=1).contiguous().view(-1, 2*c, h, w)) # (b x 2) x (2c) x h x w


        for ii in range(len(raw_fvisu)):
            corr_feat[ii] = self.corr_conv[ii](corr_feat[ii])
            corr_feat[ii] = F.normalize(corr_feat[ii], p=2, dim=1)

        ## Language Module
        if self.lstm:
            # max_len = (word_id != 0).sum(1).max().data[0]
            max_len = (word_id != 0).sum(1).max().item()
            word_id = word_id[:, :max_len]
            raw_flang, context, embedded = self.textmodel(word_id)
        else:
            all_encoder_layers, _ = self.textmodel(word_id, \
                token_type_ids=None, attention_mask=word_mask)
            ## Sentence feature at the first position [cls]  
            raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
                 + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
            ## fix bert during training
            raw_flang = raw_flang.detach()
        flang = self.mapping_lang(raw_flang)
        
        flang = F.normalize(flang, p=2, dim=1)

        flangvisu = []
        coord_list = []
        for ii in range(len(fvisu)):
            flang_tile = flang.view(flang.size(0), flang.size(1), 1, 1).\
                repeat(1, 1, fvisu[ii].size(2), fvisu[ii].size(3))
            if self.coordmap:
                coord = generate_coord(batch_size, fvisu[ii].size(2), fvisu[ii].size(3))
                coord_list.append(coord)
                flangvisu.append(torch.cat([corr_feat[ii], flang_tile, coord], dim=1))
            else:
                flangvisu.append(torch.cat([corr_feat[ii], flang_tile], dim=1))
        ## fcn

        
        intmd_fea, outbox = [], []
        for ii in range(len(fvisu)):
            intmd_fea.append(self.fcn_emb._modules[str(ii)](flangvisu[ii]))
            outbox.append(self.fcn_out._modules[str(ii)](intmd_fea[ii]))
        
        # apply subject attention here
        # max_len = (word_id != 0).sum(1).max().item()
        # word_id = word_id[:, :max_len]
        # context, embedded = self.textmodel(word_id)

        # subject attention
        









        
        sub_attn, flang_attn = self.sub_attn(context, embedded, word_id) # batchsize x 512
        flang_attn = F.normalize(flang_attn, p=2, dim=1) # batchsize x 512
        flang_attn = flang_attn.unsqueeze(2) # batchsize x 512 x 1
        flang_attn = flang_attn.unsqueeze(2) # batchsize x 512 x 1 x 1

        sim_score = []
        for ii in range(len(fvisu)):
      
            score = flang_attn * corr_feat[ii] # batchsize x 512 x H x W
            score = torch.sum(score, dim=1) # batchsize x H x W
            sim_score.append(score) 


        # apply the subject similarity score to the objectness score
        # outbox: [batchsize x (3 x 5) x h x w]
        # sim_score: [batchsize x h1 x w1, batchsize x h2 x w2, batchsize x h3 x w3]
        obj_score = []
        only_obj = []
        max_outbox_conf = []
        min_outbox_conf = []
        for ii in range(len(fvisu)):
            # reshape the outbox and then apply the subject similarity score
            batch, dummy, h, w = outbox[ii].size()
            outbox[ii] = outbox[ii].view(batch, 3, 5, h, w)

            obj_score.append(outbox[ii][:,:,4,:,:].mean(dim=1) * sim_score[ii]) # batchsize x h x w
            only_obj.append(outbox[ii][:,:,4,:,:].mean(dim=1))
            outbox[ii] = outbox[ii].view(batch, 15, h, w)
        # obj_score: [batchsize x h1 x w1, batchsize x h2 x w2, batchsize x h3 x w3]
        # then we consider location 
        # location attention
        loc_attn, flang_loc_attn = self.loc_attn(context, embedded, word_id) # batchsize x 512
        flang_loc_attn = F.normalize(flang_loc_attn, p=2, dim=1) # batchsize x 512 

        # calculate the location relation representation
        # coord_list: [batchsize x 8 x h1 x w1, batchsize x 8 x h2 x w2 , batchsize x 8 x h3 x w3]
        coord_map_list = []
        obj_score_list = []
        for ii in range(len(fvisu)):
            batch, c, h, w= coord_list[ii].size()
            coord_map_list.append(coord_list[ii].contiguous().view(batch, c, h*w).permute(0,2,1)) # batchsize x (h*w) x c
            obj_score_list.append(obj_score[ii].contiguous().view(batch, h*w)) # batchsize x (h*w) 
        coord_map = torch.cat(coord_map_list, dim=1) # batchsize x all_position x 8
        obj_map = torch.cat(obj_score_list, dim=1) # batchsize x all_position 
        obj_map = F.normalize(obj_map, p=2, dim=1) # normalize the obj map

        # embedding the coord_map
        coord_map = coord_map.contiguous().view(-1, 8)
        coord_embedding = self.loc_embedding(coord_map) # (batchsize x all_position) x 8
        embdim = coord_embedding.size(-1)
        coord_embedding = coord_embedding.contiguous().view(batch_size, -1, embdim) # batchsize x all_position x 8
        
        # normalize the embedding
        coord_embedding = F.normalize(coord_embedding, p=2, dim=2) # batchsize x all_position x 8

        # learn relation by matrix multiplication
        coord_rel_embedding = torch.bmm(coord_embedding, coord_embedding.permute(0,2,1)) # batchsize x all_position x all_position
        coord_rel_embedding = coord_rel_embedding * obj_map.unsqueeze(1) # batchsize x all_position x all_position

        coord_rel_embedding = coord_rel_embedding.contiguous().view(-1, 1344)
        coord_rel_embedding = self.loc_text_embedding(coord_rel_embedding) # (batchsize x all_position) x 512
        embdim = coord_rel_embedding.size(-1)
        coord_rel_embedding = coord_rel_embedding.contiguous().view(batch_size, -1, embdim) # batchsize x all_position x 512
        coord_rel_embedding = coord_rel_embedding.permute(0,2,1)
        coord_rel_embedding = F.normalize(coord_rel_embedding, p=2, dim=1) # note the normalization is along the channel

        # calculate the location score
        # coord_rel_embedding: batchsize x 512 x all_position, flang_loc_attn: batchsize x 512
        loc_score_map = coord_rel_embedding * flang_loc_attn.unsqueeze(-1)
        loc_score_map = torch.sum(loc_score_map, dim=1) # batchsize x all_position

        # normalization to [0,1]
        loc_score_map = (loc_score_map - loc_score_map.min(dim=1)[0].unsqueeze(1)) / (loc_score_map.max(dim=1)[0].unsqueeze(1) - loc_score_map.min(dim=1)[0].unsqueeze(1) + 1e-6)
        #loc_score_map = loc_score_map * obj_map

        # reshape the loc_score to list-style 
        # loc_score_map: batchsize x (all the positions) 
        
        # then we split the score into multiple levels
        loc_score = []
        s = 0
        for ii in range(len(fvisu)):
            e = s + fvisu[ii].size(2) * fvisu[ii].size(3)
            # print('s,e:',s,e)
            loc_score.append(loc_score_map[:,s:e].contiguous().view(-1, fvisu[ii].size(2), fvisu[ii].size(3)))
            s = e

        for ii in range(len(fvisu)):
            sim_score_ = sim_score[ii].unsqueeze(1) # batchsize x 1 x h x w
            loc_score_ = loc_score[ii].unsqueeze(1) # batchsize x 1 x h x w

            # reshape the outbox and then apply the subject similarity score and location score
            batch, dummy, h, w = outbox[ii].size()
            outbox[ii] = outbox[ii].view(batch, 3, 5, h, w)
            outbox[ii][:,:,4,:,:] = outbox[ii][:,:,4,:,:].clone() * sim_score_ * loc_score_

            outbox[ii] = outbox[ii].view(batch, 15, h, w)

       
        #------------------------------Crossmodal--------------------------------
        vit_feature=fvisu[0]
    
        vit_feature=vit_feature.flatten(-2)
        
        vit_feature=F.normalize(vit_feature,dim=2)

        lag_feature= F.interpolate(context, scale_factor=0.5)
        lag_feature = F.normalize(lag_feature, dim=1)

        lag_vit_map=torch.bmm(lag_feature, vit_feature)
        lag_vit_map=self.feature_map(lag_vit_map)

        vit_posit,lag_posit,neg_cross=Crossmodal_corrspondence(lag_feature,vit_feature,lag_vit_map)


 
        #--------------------------------------------------------------------------

        


        if self.training:
            return outbox, sim_score, loc_score, corr_feat, flang_attn,frame_feature,corrspendence_feature,neg_feature,vit_posit,lag_posit,neg_cross
            
        else:
            return outbox, sim_score, loc_score, only_obj
            
