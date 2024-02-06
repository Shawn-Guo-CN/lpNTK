import pickle
from typing import List, Tuple, OrderedDict


class Cluster(object):
    def __init__(self, head:int, sample_od:OrderedDict) -> None:
        self.head = head # the sample index of the head of the cluster
        self.sample_od = sample_od
        # an element in the sample list is a K-V pair of
        # (Key: idx of a sample, Value: its similarity to the head)

    def add(self, k:int, v:float):
        self.sample_od[k] = v

    def remove(self, k:int):
        del self.sample_od[k]
        
    def get_sample_list(self):
        return self.sample_od.keys()

    def __del__(self):
        del self.head
        del self.sample_od
        del self

    @staticmethod
    def save(cluster_list:list, path:str) -> None:
        result_list = []
        for cluster in cluster_list:
            result_list.append({'head': cluster.head,
                                'sample_od': cluster.sample_od
                              })
        with open(path, 'wb') as f:
            pickle.dump(result_list, f)

    @staticmethod
    def load(path:str) -> list:
        cluster_list = []
        with open(path, 'rb') as f:
            result_list = pickle.load(f)
        for result in result_list:
            cluster_list.append(Cluster(result['head'], result['sample_od']))
        return cluster_list
