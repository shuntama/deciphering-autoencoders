import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def clustering_kmeans(image_folder, n_clusters):
    # load images
    files = os.listdir(image_folder)
    files = sorted(files)
    images = []
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(os.path.join(image_folder, file))
            images.append(np.array(image))

    # save for display
    original_images = images.copy()

    # preprocessing
    images = np.array(images).reshape((len(images), -1))

    scaler = StandardScaler()
    images = scaler.fit_transform(images)

    # dim reduction with PCA
    pca = PCA(n_components=0.95)
    images_pca = pca.fit_transform(images)

    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(images_pca)

    cluster = kmeans.predict(images_pca)
    print('cluster counts:', np.bincount(cluster))

    # display
    """
    for i in range(n_clusters):
        images_in_cluster = np.array(original_images)[cluster == i]  # in i-th cluster
        images_in_cluster = np.random.permutation(images_in_cluster)
        plt.figure(figsize=(8, 8))
        for j in range(64):  # display 64 images from each cluster
            ax = plt.subplot(8, 8, j + 1)
            plt.imshow(images_in_cluster[j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    """
    return cluster


def gen_masks(filename, n_samples, dims, n_actives, cluster=None):
    np.random.seed(123)

    masks = []
    for j in range(n_samples):
        mask = []

        if cluster is None:
            for i in range(len(dims)):
                x = np.zeros(dims[i], dtype=bool)
                true_positions = np.random.choice(dims[i], n_actives[i], replace=False)
                x[true_positions] = True
                mask.append(x)

        else:  # cluster exists
            for i in range(len(dims)):
                if i == 0:
                    dim_c = dims[i] // (np.max(cluster) + 1)
                    x = np.zeros(dim_c, dtype=bool)
                    true_positions = np.random.choice(dim_c, n_actives[i], replace=False)
                    x[true_positions] = True

                    xl = np.zeros(dim_c * cluster[j], dtype=bool)
                    xr = np.zeros(dim_c * (np.max(cluster) - cluster[j]), dtype=bool)
                    x = np.concatenate([xl, x, xr])
                else:
                    x = np.zeros(dims[i], dtype=bool)
                    true_positions = np.random.choice(dims[i], n_actives[i], replace=False)
                    x[true_positions] = True
                mask.append(x)

        masks.append(mask)

    np.savez_compressed(filename, *masks, allow_pickle=True)


# parameters
n_samples = 50000
dims = [128, 256, 512]
n_actives = [1, 4, 16]

clustering = True
image_folder = './datasets/cifar_train/'
n_clusters = 32


# main
if clustering:
    cluster = clustering_kmeans(image_folder, n_clusters)
else:
    n_clusters = 1
    cluster = None
filename = 'masks_N{}_nf{}_na{}-{}-{}_nc{}.npz'.format(
            n_samples, dims[0], n_actives[0], n_actives[1], n_actives[2], n_clusters)
gen_masks(filename, n_samples, dims, n_actives, cluster)
