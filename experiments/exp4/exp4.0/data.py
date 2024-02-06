import numpy as np
from scipy.spatial import distance, distance_matrix
from scipy.special import softmax
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data

def get_xypp(n_data, k_class, x_dim, mu_vec, noise):
    y_all = np.random.randint(0,k_class,[n_data,1]).astype(np.float32)
    
    mu_true = np.zeros((n_data, x_dim))
    for i in range(n_data):
      mu_true[i,:] = mu_vec[y_all[i].astype(np.int),:]

    x_all = mu_true + np.random.randn(n_data, x_dim) * np.sqrt(noise)
    prior_all = np.exp((np.linalg.norm((x_all - mu_true), axis=1)**2) * \
                (-0.5/noise)) /(np.sqrt(2*np.pi*noise))

    logits = np.zeros((n_data, k_class))
    mu_vec_all = np.tile(mu_vec,(n_data,1,1))
    for k in range(k_class):
      logits[:,k] = \
          np.linalg.norm(x_all - mu_vec_all[:,k,:], axis=1)**2*(-0.5/noise)

    posterior_all = softmax(logits, 1)
    return x_all, y_all[:, 0], prior_all, posterior_all


def filter_xypp(x, y, prior, posterior, mu_vec, percentage=1.0, strategy='all'):
    n_data = x.shape[0]
    if strategy == 'all':
        return x, y, prior, posterior
    elif strategy == 'uniform':
        idx = np.random.choice(range(n_data), 
                               size=int(percentage * n_data), 
                               replace=False
                              )
        return x[idx], y[idx], prior[idx], posterior[idx]
    elif strategy == 'reverse_prior':
        idx = np.random.choice(range(n_data), 
                               size=int(percentage * n_data), 
                               replace=False,
                               p=(1/prior) / (1/prior).sum()
                              )
        return x[idx], y[idx], prior[idx], posterior[idx]
    elif strategy == 'cosine2mode':
        mode_vec = mu_vec[y.astype(int)]
        cos_dis = np.sum(x*mode_vec, axis=1) / \
                 (np.linalg.norm(x, axis=1) * np.linalg.norm(mode_vec, axis=1))
        idx = np.argsort(cos_dis) # ascending order!
        idx = idx[:int(percentage * n_data)]
        return x[idx], y[idx], prior[idx], posterior[idx]
    elif strategy == 'euclid2mode':
        mode_vec = mu_vec[y.astype(int)]
        euc_dis = np.linalg.norm(x - mode_vec, axis=1)
        idx = np.argsort(euc_dis)
        idx = idx[int(percentage * n_data):]
        return x[idx], y[idx], prior[idx], posterior[idx]
    else:
        raise ValueError('Unknown strategy: {}'.format(strategy))


def plot_x(x_all, x_uni, x_rev, x_cos, x_euc, colourcode, tsne_perplex=20):
    x_dim = x_all.shape[1]
    if x_dim == 2:
        x_all = x_all
        x_uni = x_uni
        x_rev = x_rev
        x_cos = x_cos
        x_euc = x_euc
    else:
        x_all = TSNE(n_components=2, perplexity=tsne_perplex).fit_transform(
                     x_all)
        x_uni = TSNE(n_components=2, perplexity=tsne_perplex).fit_transform(
                     x_uni)
        x_rev = TSNE(n_components=2, perplexity=tsne_perplex).fit_transform(
                     x_rev)
        x_cos = TSNE(n_components=2, perplexity=tsne_perplex).fit_transform(
                     x_cos)
        x_euc = TSNE(n_components=2, perplexity=tsne_perplex).fit_transform(
                     x_euc)
    fig, ax = plt.subplots(1, 5, figsize=(40, 7.5))
    ax1 = plt.subplot(1, 5, 1)
    ax1.scatter(x_all[:,0], x_all[:,1], c=colourcode['all'], label='all')
    ax1.title.set_text('all')
    ax2 = plt.subplot(1, 5, 2)
    ax2.scatter(x_uni[:,0], x_uni[:,1], 
                c=colourcode['uniform'], label='uniform')
    ax2.title.set_text('uniform')
    ax3 = plt.subplot(1, 5, 3)
    ax3.scatter(x_rev[:,0], x_rev[:,1], 
                c=colourcode['reverse_prior'], label='reverse_prior')
    ax3.title.set_text('reverse_prior')
    ax4 = plt.subplot(1, 5, 4)
    ax4.scatter(x_cos[:,0], x_cos[:,1], 
                c=colourcode['cosine2mode'], label='cosine2mode')
    ax4.title.set_text('cosine2mode')
    ax5 = plt.subplot(1, 5, 5)
    ax5.scatter(x_euc[:,0], x_euc[:,1], 
                c=colourcode['euclid2mode'], label='euclid2mode')
    ax5.title.set_text('euclid2mode')
    if x_dim == 2:
        plt.setp(ax, xlim=(-2, 2), ylim=(-2, 2))
    else:
        plt.setp(ax, xlim=(-70, 70), ylim=(-70, 70))
    return fig


def convert2dataloader(x, y, batch_size=256):
    dataset = Data.TensorDataset(torch.from_numpy(x).type(torch.FloatTensor), 
                                 torch.from_numpy(y).type(torch.LongTensor)
                                )
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader