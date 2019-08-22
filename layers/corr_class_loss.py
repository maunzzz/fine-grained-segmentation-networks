import torch
import torch.nn.functional as F
from torch import nn
import sys


class CorrClassLoss(nn.Module):
    def __init__(self, input_size, ignore_indices=None):
        super(CorrClassLoss, self).__init__()

        self.input_size = input_size
        self.loss = nn.CrossEntropyLoss(ignore_index=255)
        self.ignore_indices = ignore_indices

    def forward(self, inputs_ref, inputs_other, inds_ref, inds_other, weights):
        inp_ref_flat = inputs_ref.view(
            [inputs_ref.size(0), inputs_ref.size(1), -1])
        inp_other_flat = inputs_other.view(
            [inputs_ref.size(0), inputs_other.size(1), -1])
        _, inputs_ref_class_flat = inp_ref_flat.max(1)
        inputs_other_class_flat = 255*torch.ones_like(inputs_ref_class_flat)

        for b in range(inputs_ref.size(0)):
            b_inds_ref = inds_ref[b]
            b_inds_other = inds_other[b]

            b_inds_ref = b_inds_ref.type(torch.int64)
            b_inds_other = b_inds_other.type(torch.int64)

            valid_inds = (b_inds_ref[0, :] < inputs_ref.size(3)) & (b_inds_ref[0, :] >= 0) & (b_inds_ref[1, :] < inputs_ref.size(2)) & (b_inds_ref[1, :] >= 0) & (
                b_inds_other[0, :] < inputs_ref.size(3)) & (b_inds_other[0, :] >= 0) & (b_inds_other[1, :] < inputs_ref.size(2)) & (b_inds_other[1, :] >= 0)

            inds_ref_this = b_inds_ref[:, valid_inds]
            inds_other_this = b_inds_other[:, valid_inds]

            n_matches = inds_ref_this.size(1)

            lin_inds_ref = inputs_ref.size(
                2)*inds_ref_this[1, :] + inds_ref_this[0, :]
            lin_inds_other = inputs_ref.size(
                2)*inds_other_this[1, :] + inds_other_this[0, :]

            inputs_other_class_flat[b,
                                    lin_inds_other] = inputs_ref_class_flat[b, lin_inds_ref]

        inputs_other_class = inputs_other_class_flat.view(
            [inputs_ref.size(0), inputs_ref.size(2), inputs_ref.size(3)])

        if self.ignore_indices is not None:
            for class_to_ignore in self.ignore_indices:
                inputs_other_class[inputs_other_class == class_to_ignore] = 255

        return self.loss(inputs_other, inputs_other_class)
