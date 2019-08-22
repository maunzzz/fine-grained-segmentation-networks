import os
import re
import numpy as np


def parse_log_clustering(file_path):
    #train_pattern = re.compile(r'^\[iter (\d+) / (\d+)\], \[train seg loss ([\.\d]+)\]\. \[lr ([\.\d]+)\]$')
    train_pattern = re.compile(
        r'^\[iter (\d+) / (\d+)\], \[train seg loss ([\.\d]+)\], \[train corr loss ([\.\d]+)\], \[train feature loss ([\.\d]+)\]\. \[lr ([\.\d]+)\]$')
    train_pattern_fallback = re.compile(
        r'^\[iter (\d+) / (\d+)\], \[train seg loss ([\.\d]+)\], \[train corr loss ([\.\d]+)\]\. \[lr ([\.\d]+)\]$')  # for backward compatability
    val_pattern = re.compile(
        r'^([a-zA-Z]+) \[iter (\d+)\], \[val loss feat ([\.\d]+)\], \[val loss out ([\.\d]+)\], \[val loss cluster ([\.\d]+)\]$')
    cluster_pattern = re.compile(r'^cluster distribution \[([\. \d]+)\]')

    n_iterations_between_clusters = None
    train_dict = {}
    train_dict['iter'] = []
    train_dict['seg_loss'] = []
    train_dict['corr_loss'] = []
    train_dict['feat_loss'] = []
    train_dict['lr'] = []
    val_dict = {}
    cluster_distributions = []

    with open(file_path) as f:
        for line in f:
            t_match = train_pattern.match(line)
            tf_match = train_pattern_fallback.match(line)
            v_match = val_pattern.match(line)
            c_match = cluster_pattern.match(line)

            if t_match:
                train_dict['iter'].append(int(t_match.group(1)))
                train_dict['seg_loss'].append(float(t_match.group(3)))
                train_dict['corr_loss'].append(float(t_match.group(4)))
                train_dict['feat_loss'].append(float(t_match.group(5)))
                train_dict['lr'].append(float(t_match.group(6)))
                if n_iterations_between_clusters is None:
                    n_iterations_between_clusters = int(t_match.group(2))

            if tf_match:
                train_dict['iter'].append(int(tf_match.group(1)))
                train_dict['seg_loss'].append(float(tf_match.group(3)))
                train_dict['corr_loss'].append(float(tf_match.group(4)))
                train_dict['feat_loss'].append(0.)
                train_dict['lr'].append(float(tf_match.group(5)))
                if n_iterations_between_clusters is None:
                    n_iterations_between_clusters = int(tf_match.group(2))

            if v_match:
                if v_match.group(1) not in val_dict.keys():
                    val_dict[v_match.group(1)] = {}
                    val_dict[v_match.group(1)]['iter'] = []
                    val_dict[v_match.group(1)]['feat'] = []
                    val_dict[v_match.group(1)]['out'] = []
                    val_dict[v_match.group(1)]['cluster'] = []

                val_dict[v_match.group(1)]['iter'].append(int(v_match.group(2)))
                val_dict[v_match.group(1)]['feat'].append(float(v_match.group(3)))
                val_dict[v_match.group(1)]['out'].append(float(v_match.group(4)))
                val_dict[v_match.group(1)]['cluster'].append(float(v_match.group(5)))
            if c_match:
                cluster_distributions.append(np.fromstring(
                    c_match.group(1), dtype=np.float, sep=' '))

    return train_dict, val_dict, cluster_distributions, n_iterations_between_clusters


if __name__ == '__main__':
    train_dict, val_dict, cluster_distributions, n_iterations_between_clusters = parse_log_clustering(
        '/home/user/example.log')
