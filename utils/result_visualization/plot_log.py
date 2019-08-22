import os
import re
import matplotlib.pyplot as plt
import numpy as np
from parse_log import parse_log_clustering


def plot_log_clustering(log_path, save_fig, show_fig):
    train_dict, val_dict, cluster_distributions, n_iterations_between_clusters = parse_log_clustering(
        log_path)

    mm = re.match(r".*-ci(\d+)-.*", log_path)
    cluster_interval = int(mm.group(1))
    cluster_occasions = list(
        range(cluster_interval, train_dict['iter'][-1], cluster_interval))

    plt.figure(1)
    plt.subplot(231)
    for cluster_occasion in cluster_occasions:
        plt.axvline(x=cluster_occasion, color='r', linestyle='--')
    plt.plot(train_dict['iter'], train_dict['seg_loss'], label='seg')
    plt.plot(train_dict['iter'], train_dict['corr_loss'], label='corr')
    plt.plot(train_dict['iter'], train_dict['feat_loss'], label='feat')
    plt.legend()
    plt.xlabel('iteration')
    plt.title('training loss')

    scatter_x = np.zeros(
        (len(cluster_distributions)*len(cluster_distributions[0])))
    scatter_y = np.zeros_like(scatter_x)
    for i, cluster_dist in enumerate(cluster_distributions):
        nval = len(cluster_dist)
        scatter_x[i*nval:(i+1)*nval] = i*n_iterations_between_clusters
        scatter_y[i*nval:(i+1)*nval] = cluster_dist

    plt.subplot(232)
    for cluster_occasion in cluster_occasions:
        plt.axvline(x=cluster_occasion, color='r', linestyle='--')

    m = max(train_dict['seg_loss'])
    if m == 0.:
        m = 1.
    plt.plot(train_dict['iter'], [
        l/m for l in train_dict['seg_loss']], label='seg')
    m = max(train_dict['corr_loss'])
    if m == 0.:
        m = 1.
    plt.plot(train_dict['iter'], [
        l/m for l in train_dict['corr_loss']], label='corr')
    m = max(train_dict['feat_loss'])
    if m == 0.:
        m = 1.
    plt.plot(train_dict['iter'], [
        l/m for l in train_dict['feat_loss']], label='feat')

    plt.legend()
    plt.xlabel('iteration')
    plt.title('training loss normalized')

    plt.subplot(233)
    plt.scatter(scatter_x, scatter_y)
    plt.xlabel('iteration')
    plt.ylabel('cluster sizes')

    plt.subplot(234)
    for cluster_occasion in cluster_occasions:
        plt.axvline(x=cluster_occasion, color='r', linestyle='--')

    if 'Corr' in val_dict:
        v = val_dict['Corr']
        plt.plot(v['iter'], v['feat'])
        plt.xlabel('iteration')
        plt.ylabel('feat')
        plt.title('validation')

        plt.subplot(235)
        for cluster_occasion in cluster_occasions:
            plt.axvline(x=cluster_occasion, color='r', linestyle='--')
        plt.plot(v['iter'], v['out'])
        plt.xlabel('iteration')
        plt.ylabel('out')
        plt.title('validation')

        plt.subplot(236)
        for cluster_occasion in cluster_occasions:
            plt.axvline(x=cluster_occasion, color='r', linestyle='--')
        plt.plot(v['iter'], v['cluster'])
        plt.xlabel('iteration')
        plt.ylabel('cluster')
        plt.title('validation')

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_fig:
        plot_path = os.path.join(os.path.dirname(
            os.path.realpath(log_path)), 'log.png')
        print('plot saved as %s' % plot_path)
        plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    log_paths = []
    log_paths.append('example.log')

    save_fig = True
    show_fig = False
    for log_path in log_paths:
        plot_log_clustering(log_path, save_fig, show_fig)
