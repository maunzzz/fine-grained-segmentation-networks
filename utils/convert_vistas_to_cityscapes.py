import os
import PIL.Image
import numpy as np
import json


def convert_folder(in_folder, out_folder, label_mapping):

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    in_files = [f for f in os.listdir(in_folder) if f.endswith('.png')]

    for in_file in in_files:
        in_full_path = os.path.join(in_folder, in_file)

        in_lbl = np.asarray(PIL.Image.open(in_full_path))

        out_lbl = 255 * np.ones_like(in_lbl)
        for k, v in label_mapping.items():
            out_lbl[in_lbl == k] = v

        out_lbl_im = PIL.Image.fromarray(out_lbl.astype(np.uint8))
        out_lbl_im.save(os.path.join(out_folder, in_file), "PNG")


mapillary_path = '/media/user/Data/mapillary-vistas-dataset_public_v1.0/'

ignore_ind = 255
map_to_cs = {0: ignore_ind, 1: ignore_ind, 2: ignore_ind, 3: 13, 4: ignore_ind, 5: ignore_ind, 6: 12, 7: ignore_ind, 8: 7, 9: ignore_ind, 10: ignore_ind, 11: ignore_ind, 12: ignore_ind,
             13: 7, 14: 7, 15: 8, 16: ignore_ind, 17: 11, 18: ignore_ind, 19: 24, 20: 25, 21: 25, 22: 25, 23: 7, 24: 7, 25: ignore_ind, 26: ignore_ind, 27: 23, 28: ignore_ind, 29: 22, 30: 21,
             31: ignore_ind, 32: ignore_ind, 33: ignore_ind, 34: ignore_ind, 35: ignore_ind, 36: ignore_ind, 37: ignore_ind, 38: ignore_ind, 39: ignore_ind, 40: ignore_ind, 41: ignore_ind, 42: ignore_ind, 43: ignore_ind, 44: 17, 45: 17, 46: ignore_ind, 47: 17, 48: 19, 49: ignore_ind, 50: 20, 51: ignore_ind, 52: 33, 53: ignore_ind, 54: 28, 55: 26, 56: ignore_ind, 57: 32, 58: 31, 59: ignore_ind, 60: ignore_ind, 61: 27, 62: ignore_ind, 63: ignore_ind, 64: ignore_ind, 65: ignore_ind}

map_training_labels = os.path.join(mapillary_path, 'training', 'labels')
cs_training_labels = os.path.join(mapillary_path, 'training', 'labels_cityscapes')

convert_folder(map_training_labels, cs_training_labels, map_to_cs)

map_validation_labels = os.path.join(mapillary_path, 'validation', 'labels')
cs_validation_labels = os.path.join(mapillary_path, 'validation', 'labels_cityscapes')

convert_folder(map_validation_labels, cs_validation_labels, map_to_cs)
