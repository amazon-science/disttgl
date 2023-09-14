import argparse
import yaml
import torch
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sampler_core import ParallelSampler, TemporalGraphBlock

class NegLinkSampler:

    def __init__(self, num_nodes, df=None, transduct=False, induct=False, num_rngs=0):
        self.random_generator = None
        self.num_rngs = num_rngs
        self.rng_list = list()
        self.num_nodes = num_nodes
        self.random_sample = True
        self.dst_nodes = None
        if df is not None:
            self.random_sample = False
            train_edge_end = df[df['ext_roll'].gt(0)].index[0]
            if transduct:
                self.dst_nodes = np.unique(df[:train_edge_end].dst.values.astype(np.int64))
            elif induct:
                self.dst_nodes = np.unique(df[train_edge_end:].dst.values.astype(np.int64))
                trans_nodes = np.unique(df[:train_edge_end].dst.values.astype(np.int64))
                self.dst_nodes = self.dst_nodes[np.logical_not(np.isin(self.dst_nodes, trans_nodes))]
            else:
                self.dst_nodes = np.unique(df.dst.values.astype(np.int64))
        self.reset_rng()

    def reset_rng(self, i=None):
        if self.num_rngs > 0:
            if i is None:
                self.random_generator = np.random.default_rng(random.randint(0, self.num_rngs - 1))
            else:
                self.random_generator = np.random.default_rng(i % self.num_rngs)
        else:
            self.random_generator = np.random.default_rng()

    def sample(self, n):
        if self.random_sample:
            return self.random_generator.integers(self.num_nodes, size=n)
        else:
            return self.random_generator.choice(self.dst_nodes, size=n)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--batch_size', type=int, default=600, help='path to config file')
    parser.add_argument('--num_thread', type=int, default=64, help='number of thread')
    args=parser.parse_args()

    df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
    g = np.load('DATA/{}/ext_full.npz'.format(args.data))
    sample_config = yaml.safe_load(open(args.config, 'r'))['sampling'][0]

    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              args.num_thread, 1, sample_config['layer'], sample_config['neighbor'],
                              sample_config['strategy']=='recent', sample_config['prop_time'],
                              sample_config['history'], float(sample_config['duration']))

    num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
    neg_link_sampler = NegLinkSampler(num_nodes)

    tot_time = 0
    ptr_time = 0
    coo_time = 0
    sea_time = 0
    sam_time = 0
    uni_time = 0
    total_nodes = 0
    unique_nodes = 0
    for _, rows in tqdm(df.groupby(df.index // args.batch_size), total=len(df) // args.batch_size):
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        tot_time += ret[0].tot_time()
        ptr_time += ret[0].ptr_time()
        coo_time += ret[0].coo_time()
        sea_time += ret[0].search_time()
        sam_time += ret[0].sample_time()
        # for i in range(sample_config['history']):
        #     total_nodes += ret[i].dim_in() - ret[i].dim_out()
        #     unique_nodes += ret[i].dim_in() - ret[i].dim_out()
        #     if ret[i].dim_in() > ret[i].dim_out():
        #         ts = torch.from_numpy(ret[i].ts()[ret[i].dim_out():])
        #         nid = torch.from_numpy(ret[i].nodes()[ret[i].dim_out():]).float()
        #         nts = torch.stack([ts,nid],dim=1).cuda()
        #         uni_t_s = time.time()
        #         unts, idx = torch.unique(nts, dim=0, return_inverse=True)
        #         uni_time += time.time() - uni_t_s
        #         total_nodes += idx.shape[0]
        #         unique_nodes += unts.shape[0]

    print('total time  : {:.4f}'.format(tot_time))
    print('pointer time: {:.4f}'.format(ptr_time))
    print('coo time    : {:.4f}'.format(coo_time))
    print('search time : {:.4f}'.format(sea_time))
    print('sample time : {:.4f}'.format(sam_time))
    # print('unique time : {:.4f}'.format(uni_time))
    # print('unique per  : {:.4f}'.format(unique_nodes / total_nodes))


    