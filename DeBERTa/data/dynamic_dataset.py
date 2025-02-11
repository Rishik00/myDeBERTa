# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 05/15/2019
#

import pdb
from torch.utils.data import Dataset
import random
import numpy as np
from bisect import bisect
from ..utils import get_logger

logger = get_logger()

__all__ = ['DynamicDataset']

class DynamicDataset(Dataset):
  def __init__(self, corpus, feature_fn, dataset_size=None, shuffle=False, **kwargs):
    self.corpus = corpus
    self.ds_len = len(self.corpus)
    logger.info(f'Total corpus examples: {self.ds_len}')
    self.feature_fn = feature_fn

    self.dataset_size = dataset_size if dataset_size else self.ds_len
    self.dataset_size = int(self.dataset_size)

    self.shuffle = shuffle
    index_buf = np.zeros(self.dataset_size, dtype=np.int32)
    shuffle_idx = np.arange(self.dataset_size, dtype=np.int32)

    if self.shuffle:
      random.seed(0)
      random.shuffle(shuffle_idx)

    self.shuffle_idx = shuffle_idx
    self.index_offset = kwargs.get('index_offset', 0)

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, idx):
    if isinstance(idx, (tuple, list)):
      idx, ext_params = idx
    else:
      ext_params = None
    
    idx = int(idx)  # Ensure idx is an integer
    idx += self.index_offset
    seed = idx
    random.seed(seed)  # Now guaranteed to be an int
    
    # Get sequence length
    example_idx = self.shuffle_idx[idx % self.dataset_size] % self.ds_len
    example = self.corpus[example_idx]  # Removed incorrect tuple indexing
    return self.feature_fn(example, ext_params=ext_params)
