# Originally from: https://github.com/facebookresearch/deepcluster
# This source code is licensed under the license found in the
# LICENSE file found in that repository (Attribution-NonCommercial 4.0 International)
# Modified

import time
import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess_features(npdata, pca=256, pca_info=None):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    if pca_info is None:
        # Apply PCA-whitening with Faiss
        pca_matrix = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        pca_matrix.train(npdata)
        assert pca_matrix.is_trained
        npdata = pca_matrix.apply_py(npdata)

        pca_A = np.transpose(faiss.vector_to_array(pca_matrix.A).reshape((pca, ndim)))
        pca_b = faiss.vector_to_array(pca_matrix.b)
        pca_info = (pca_A, pca_b)
    else:
        npdata = np.dot(npdata, pca_info[0]) + pca_info[1]

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata, pca_info


def run_kmeans(x, nmb_clusters, verbose=False, use_gpu=True):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    if use_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
    else:
        index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    centroids = faiss.vector_to_array(clus.centroids).reshape((nmb_clusters, d))  # Also return centroids!
    losses = faiss.vector_to_array(clus.obj)

    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1], centroids, index


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans:
    def __init__(self, k):
        self.k = k
        self.index = None

    def set_index(self, centroids):
        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(centroids)
        self.index = index

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb, _ = preprocess_features(data)

        # cluster the data
        I, loss, _ = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss

    def cluster_imfeatures(self, data, verbose=False, use_gpu=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster

            returns cluster indices for each feature                
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb, pca_info = preprocess_features(data)

        # cluster the data
        I, loss, centroids, index = run_kmeans(xb, self.k, verbose, use_gpu=use_gpu)

        self.index = index

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return I, loss, centroids, pca_info

    def assign(self, data):
        _, I = self.index.search(data, 1)
        return I
