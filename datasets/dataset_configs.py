import os
from utils.misc import get_global_opts
global_opts = get_global_opts()


class DatasetConfig(object):
    def __init__(self):
        self.train_im_folder = None
        self.train_seg_folder = None
        self.val_im_folder = None
        self.val_seg_folder = None
        self.test_im_folder = None
        self.test_seg_folder = None
        self.id_to_trainid = False
        self.im_ext = '.jpg'
        self.n_classes = 19


class CityscapesConfig(DatasetConfig):
    def __init__(self):
        super(CityscapesConfig, self).__init__()
        root = global_opts['cityscapes_path']

        self.train_im_folder = os.path.join(root, 'leftImg8bit', 'train')
        self.train_seg_folder = os.path.join(root, 'gtFine', 'train')

        self.train_extra_im_folder = os.path.join(root, 'leftImg8bit_trainextra' 'leftImg8bit', 'train_extra')
        self.train_extra_seg_folder = os.path.join(root, 'gtCoarse', 'gtCoarse', 'train_extra')

        self.val_im_folder = os.path.join(root, 'leftImg8bit', 'val')
        self.val_seg_folder = os.path.join(root, 'gtFine', 'val')
        self.test_im_folder = os.path.join(root, 'leftImg8bit', 'test')

        self.im_file_ending = 'leftImg8bit.png'
        self.seg_file_ending = 'gtFine_labelIds.png'

        ignore_label = 255
        self.ignore_label = ignore_label
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


class VistasConfig(DatasetConfig):
    def __init__(self, use_subsampled_validation_set=False, use_cityscapes_classes=True, include_roadmarks=False):
        super(VistasConfig, self).__init__()
        root = global_opts['vistas_path']

        self.train_im_folder = os.path.join(root, 'training', 'images')
        if use_cityscapes_classes:
            if include_roadmarks:
                self.train_seg_folder = os.path.join(root, 'training', 'labels_cityscapes_with_roadmarkings')
            else:
                self.train_seg_folder = os.path.join(root, 'training', 'labels_cityscapes')
        else:
            self.train_seg_folder = os.path.join(root, 'training', 'labels')

        if use_subsampled_validation_set:
            self.val_im_folder = os.path.join(root, 'validation_subsampled', 'images')
        else:
            self.val_im_folder = os.path.join(root, 'validation', 'images')

        if use_cityscapes_classes:
            if include_roadmarks:
                self.val_seg_folder = os.path.join(root, 'validation', 'labels_cityscapes_with_roadmarkings')
                self.n_classes = 20
            else:
                self.val_seg_folder = os.path.join(root, 'validation', 'labels_cityscapes')
        else:
            self.val_seg_folder = os.path.join(root, 'validation', 'labels')
            self.n_classes = 66

        self.im_file_ending = '.jpg'
        self.seg_file_ending = '.png'

        if use_cityscapes_classes:
            ignore_label = 255
            self.ignore_label = ignore_label
            if include_roadmarks:
                self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                                      3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                                      7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                                      14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                                      18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                                      28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18, 50: 19}
            else:
                self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                                      3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                                      7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                                      14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                                      18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                                      28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18, 50: 0}
        else:
            self.ignore_label = 65
            self.id_to_trainid = None


class RobotcarConfig(DatasetConfig):
    def __init__(self, use_vistas_classes=False):
        root = global_opts['robotcar_root_path']

        super(RobotcarConfig, self).__init__()

        if use_vistas_classes:
            anno_folder_name = 'annos_vistas'
            self.n_classes = 66
        else:
            anno_folder_name = 'annos'

        self.train_im_folder = os.path.join(root, 'segmented_images_201810', 'training', 'imgs')
        self.train_seg_folder = os.path.join(root, 'segmented_images_201810', 'training', anno_folder_name)

        self.val_im_folder = os.path.join(root, 'segmented_images_201810', 'validation', 'imgs')
        self.val_seg_folder = os.path.join(root, 'segmented_images_201810', 'validation', anno_folder_name)

        self.test_im_folder = os.path.join(root, 'segmented_images_201810', 'testing', 'imgs')
        self.test_seg_folder = os.path.join(root, 'segmented_images_201810', 'testing', anno_folder_name)

        self.vis_test_im_folder = os.path.join(root, 'oxford-visual-test-images')

        self.im_file_ending = '.png'
        self.seg_file_ending = '.png'

        ignore_label = 255
        self.ignore_label = ignore_label
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        self.correspondence_path = global_opts['robotcar_corr_path']
        self.correspondence_im_path = global_opts['robotcar_im_path']
        self.correspondence_train_list_file = os.path.join(root, 'corr_train.txt')
        self.correspondence_val_list_file = os.path.join(root, 'corr_val.txt')
        self.reference_image_list = os.path.join(root, 'reference_image_names.txt')
        self.reference_feature_poitions = global_opts['robotcar_corr_path'] + '_reference_feature_positions'


class CmuConfig(DatasetConfig):
    def __init__(self, use_vistas_classes=False):
        root = global_opts['cmu_root_path']

        super(CmuConfig, self).__init__()

        if use_vistas_classes:
            anno_folder_name = 'annos_vistas'
            self.n_classes = 66
        else:
            anno_folder_name = 'annos'

        self.train_im_folder = os.path.join(root, 'segmented_images', 'training', 'imgs')
        self.train_seg_folder = os.path.join(root, 'segmented_images', 'training', anno_folder_name)

        self.val_im_folder = os.path.join(root, 'segmented_images', 'validation', 'imgs')
        self.val_seg_folder = os.path.join(root, 'segmented_images', 'validation', anno_folder_name)

        self.test_im_folder = os.path.join(root, 'segmented_images', 'testing', 'imgs')
        self.test_seg_folder = os.path.join(root, 'segmented_images', 'testing', anno_folder_name)

        self.vis_test_im_folder = os.path.join(root, 'cmu-visual-test-images')

        self.im_file_ending = '.png'
        self.seg_file_ending = '.png'

        ignore_label = 255
        self.ignore_label = ignore_label
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        self.correspondence_path = global_opts['cmu_corr_path']
        self.correspondence_im_path = global_opts['cmu_im_path']
        self.correspondence_train_list_file = os.path.join(root, 'corr_train.txt')
        self.correspondence_val_list_file = os.path.join(root, 'corr_val.txt')
        self.reference_image_list = os.path.join(root, 'reference_image_names.txt')
        self.reference_feature_poitions = global_opts['cmu_corr_path'] + '_reference_feature_positions'

        self.point_cloud_root = os.path.join(root, 'data_for_3d_segmentation_training')
