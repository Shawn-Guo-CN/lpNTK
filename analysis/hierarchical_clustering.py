import numpy as np
import math
import weakref
from collections import defaultdict


class UnionTracker(object):
    def __init__(self):
        self.linkage_matrix = []
    
    def add(self, idx1, idx2, norm, tgt_idx):
        '''Create and append a new Union Entry to the linkage list
        
        Args:
            idx1 (int): index of the first child of the merged union
            idx2 (int): index of the second child of the merged union
            norm (float): smaller norm of the two children unions
            tgt_idx (int): index of the merged union
        '''
        self.linkage_matrix.append([idx1, idx2, norm, tgt_idx])

    def get_linkage_matrix(self):
        return np.array(self.linkage_matrix)

    def save(self, path):
        np.save(path, np.array(self.linkage_matrix))

    def load(self, path):
        self.linkage_matrix = np.load(path)

        
class Cluster(object):
    count = 0            # Count the number of clusters  
    max_norm = -1        # Track max norm of the clusters
    min_norm = math.inf  # Track min norm of the clusters
    mean_norm = 0        # Track mean norm of the clusters
    _instances = defaultdict()

    def __init__(self, **kwargs):
        self.idx = Cluster.count
        Cluster.count += 1

        self.type = kwargs['type']
        self.norm = None
        self.sample_idx_list = None
        self.lchild = None
        self.rchild = None

        if self.type == 'leaf':
            self.sample_idx_list = [kwargs['sample_idx']]
            self.norm = -1
        elif self.type == 'internal':
            self.lchild = kwargs['lchild']
            self.rchild = kwargs['rchild']
            self.norm = kwargs['norm']
            self.sample_idx_list = kwargs['lchild_sample_idx'] + kwargs['rchild_sample_idx']
        else:
            raise ValueError(f'Invalid type of cluster: {self.type}')
    
        if self.norm > Cluster.max_norm:
            Cluster.max_norm = self.norm
        if self.norm < Cluster.min_norm:
            Cluster.min_norm = self.norm

        Cluster.mean_norm = (Cluster.mean_norm * (Cluster.count - 1) + self.norm) / Cluster.count
        self._instances[self.idx] = weakref.ref(self)

    def __del__(self):
        ''' Destructor for the Cluster Object'''
        Cluster._instances.pop(self.idx, None)


