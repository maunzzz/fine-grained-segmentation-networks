from math import sqrt
from train_with_clustering import train_with_clustering_experiment
# Template for running several experiments
args = {
    # general training settings
    'train_batch_size': 1,
    # probability of propagating error for reference image instead of target imare (set to None to use both)
    'fraction_reference_bp': 0.5,
    'lr': 1e-4 / sqrt(16 / 1),
    'lr_decay': 1,
    'max_iter': 60000,
    'weight_decay': 1e-4,
    'momentum': 0.9,

    # starting network settings
    'startnet': 'vis',  # specify full path or set to 'vis' for network trained with vistas + cityscapes or 'cs' for network trained with cityscapes
    'use_original_base': False,  # must be true if starting from classification network

    # set to '' to start training from beginning and 'latest' to use last checkpoint
    'snapshot': 'latest',

    # dataset settings
    'corr_set': 'cmu',  # 'cmu', 'rc', 'both' or 'none'
    'max_features_per_image': 500,  # dont set to high (RAM runs out)

    # clustering settings
    'n_clusters': 100,
    'cluster_interval': 10000,

    # loss settings
    'corr_loss_weight': 1,  # was 1
    'seg_loss_weight': 1,  # was 1
    'feature_hinge_loss_weight': 0,  # was 0

    # validation settings
    'val_interval': 2500,
    'feature_distance_measure': 'L2',

    # misc
    'chunk_size': 50,
    'print_freq': 10,
    'stride_rate': 2 / 3.,
    'n_workers': 1,  # set to 0 for debugging
}


# Main Experiments CMU
args['n_clusters'] = 20
train_with_clustering_experiment(args.copy())
args['n_clusters'] = 100
train_with_clustering_experiment(args.copy())
args['n_clusters'] = 200
train_with_clustering_experiment(args.copy())
args['n_clusters'] = 1000
train_with_clustering_experiment(args.copy())

# Main Experiments RC
args['corr_set'] = 'rc'
args['n_clusters'] = 20
train_with_clustering_experiment(args.copy())
args['n_clusters'] = 100
train_with_clustering_experiment(args.copy())
args['n_clusters'] = 200
train_with_clustering_experiment(args.copy())
args['n_clusters'] = 1000
train_with_clustering_experiment(args.copy())
