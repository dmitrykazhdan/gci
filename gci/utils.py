import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import shutil


def compute_tsne_features(features_np):
    tsne = TSNE(n_components=2)
    tsne_features = tsne.fit_transform(features_np)
    return tsne_features


def convert_one_hot_arr_to_categorical(one_hot_arr):
    items_tuple_list = [tuple(i) for i in one_hot_arr]
    unique_items_tuple_list = list(set(items_tuple_list))
    sorted_unique_items_tuple_list = sorted(unique_items_tuple_list, reverse=True)

    categorical_tuple_index_array = [sorted_unique_items_tuple_list.index(tuple(i)) for i in one_hot_arr]
    categorical_tuple_index_array = np.array(categorical_tuple_index_array)
    return categorical_tuple_index_array


def create_item_to_id_dict(items):
    item_to_id_dict = {}

    for item in items:
        if item not in item_to_id_dict.keys():
            item_to_id_dict[item] = len(item_to_id_dict.keys()) + 1

    return item_to_id_dict


def setup_folder_from_scratch(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def generate_or_load_torch_train_test_split(dataset, n_train, n_val, base_path="./"):
    paths = ['train', 'val', 'test']
    idx_arrs = []

    full_train_path = os.path.join(base_path, "train.npy")
    if os.path.exists(full_train_path):
        for i, pth in enumerate(paths):
            full_path = os.path.join(base_path, f"./{pth}.npy")
            pth_data = np.load(full_path)
            idx_arrs.append(pth_data)

        train_idxs, val_idxs, test_idxs = idx_arrs[0], idx_arrs[1], idx_arrs[2]

    else:
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)

        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:n_train+n_val]
        test_idxs = idxs[(n_train+n_val):]

        for idx_data, pth in zip([train_idxs, val_idxs, test_idxs], paths):
            full_path = os.path.join(base_path, f"./{pth}.npy")
            np.save(full_path, idx_data)


    train_idxs = [int(i) for i in list(train_idxs)]
    val_idxs = [int(i) for i in list(val_idxs)]
    test_idxs = [int(i) for i in list(test_idxs)]

    train_set, valid_set, test_set = torch.utils.data.Subset(dataset, train_idxs), \
                                     torch.utils.data.Subset(dataset, val_idxs), \
                                     torch.utils.data.Subset(dataset, test_idxs)

    return train_set, valid_set, test_set
