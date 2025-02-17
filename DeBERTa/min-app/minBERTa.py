from glob import glob
from collections import OrderedDict,defaultdict
from collections.abc import Sequence
from bisect import bisect
import copy
import math
from scipy.special import softmax
import numpy as np
import pdb
import os
import sys
import csv

import random
import torch
import re
import shutil
import ujson as json
from torch.utils.data import DataLoader

class MaskedLanguageModel(torch.nn.Module):
    pass

class ReplacedTokenDetectionModel(torch.nn.Module):
    pass

class RTDModel(torch.nn.Module):
    def __init__(self, config, *wargs, **kwargs):
        gconfig = config.generator
        dconfig = config.discriminator

        self.config = config
        self.gen = MaskedLanguageModel()
        self.discrim = ReplacedTokenDetectionModel()

        self.share_embeds = config.embedding_sharing
        if self.share_embeds == 'gdes':
            pass

        self.register_discrim_fw_hook()

    def topk(self, logits, topk=1, start=0, temp=1):
        pass

    def make_electra_data(self, input_data, temp=1, rand=None):
        new_data = input_data.copy()
        if rand is None: rand = random
        gen = self.generator_fw(**new_data)
        lm_logits = gen['logits']
        lm_labels = input_data['labels']
        lm_loss = gen['loss']

        mask_index = (lm_labels.view(-1)>0).nonzero().view(-1)
        gen_pred = torch.argmax(lm_logits, dim=1).detach().cpu().numpy()
        topk_labels, top_p = self.topk_sampling(lm_logits, topk=1, temp=temp)
        
        top_ids = torch.zeros_like(lm_labels.view(-1))
        top_ids.scatter_(index=mask_index, src=topk_labels.view(-1).int(), dim=-1)
        top_ids = top_ids.view(lm_labels.size())
        new_ids = torch.where(lm_labels>0, top_ids, input_data['input_ids'])
        new_data['input_ids'] = new_ids.detach()
        
        return new_data, lm_loss, gen

    def register_discrim_fw_hook(self):
        pass
    
    def generator_fw(self, **kwargs):
        return self.gen(**kwargs)

    def discriminator_fw(self, **kwargs):
        return self.discrim(**kwargs)
    
    def forward(self, **kwargs):
        return self.generator_fw(**kwargs)
    






