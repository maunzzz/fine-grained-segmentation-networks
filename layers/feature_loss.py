import torch
import torch.nn.functional as F
from torch import nn
import math


class FeatureLoss(nn.Module):
    def __init__(self, input_size, size_average=True, n_not_matching=5, min_distance_for_missmatches=200, non_match_weight=None, loss_type='L1', feat_dist_threshold_nomatch=0.5, feat_dist_threshold_match=0.0):

        self.min_distance_for_missmatches = min_distance_for_missmatches  # in pixels
        self.n_not_matching = n_not_matching
        self.loss_type = loss_type  # L1, L2, hinge or KL
        self.size_average = size_average
        self.input_size = input_size
        self.feat_dist_threshold_nomatch = feat_dist_threshold_nomatch  # per feature!
        self.feat_dist_threshold_match = feat_dist_threshold_match  # per feature!
        # self.class_inds_to_remove = class_inds_to_remove #This feature have been moved to an external function

        if non_match_weight is None:
            self.non_match_weight = self.n_not_matching
        else:
            self.non_match_weight = non_match_weight

        super(FeatureLoss, self).__init__()

    def forward(self, inputs_ref, inputs_other, inds_ref, inds_other, weights):
        if weights is None:
            weights = []
            for i_ref in inds_ref:
                weights.append(torch.ones(
                    [i_ref.size(1), 1], dtype=inputs_ref.dtype, device=inputs_ref.device))

        max_distance = math.sqrt((inputs_ref.size(2)**2+inputs_ref.size(3)**2))

        if self.loss_type == 'L1':
            feat_dist_thresh_nomatch = self.feat_dist_threshold_nomatch * \
                inputs_ref.size(1)
            feat_dist_thresh_match = self.feat_dist_threshold_match * \
                inputs_ref.size(1)
        elif self.loss_type == 'L2':
            feat_dist_thresh_nomatch = self.feat_dist_threshold_nomatch**2 * \
                inputs_ref.size(1)
            feat_dist_thresh_match = self.feat_dist_threshold_match**2 * \
                inputs_ref.size(1)
        elif self.loss_type == 'hingeC' or self.loss_type == 'hingeF':
            feat_dist_thresh_nomatch = self.feat_dist_threshold_nomatch
            feat_dist_thresh_match = self.feat_dist_threshold_match
            # also L2-normalize features
            inputs_ref = F.normalize(inputs_ref, dim=1)
            inputs_other = F.normalize(inputs_other, dim=1)
        elif self.loss_type == 'KL':
            # also L2-normalize features
            inputs_ref = F.softmax(inputs_ref, dim=1)
            inputs_other = F.softmax(inputs_other, dim=1)

        inp_ref_flat = inputs_ref.view(
            [inputs_ref.size(0), inputs_ref.size(1), -1])
        inp_other_flat = inputs_other.view(
            [inputs_ref.size(0), inputs_other.size(1), -1])

        loss = 0.0
        for b in range(inputs_ref.size(0)):
            b_inds_ref = inds_ref[b]
            b_inds_other = inds_other[b]
            b_weights = weights[b]

            # rescale correspondence indices to feature size
            if self.input_size != [inputs_ref.size(2), inputs_ref.size(3)]:
                b_inds_ref = b_inds_ref.type(torch.float32)
                b_inds_other = b_inds_other.type(torch.float32)

                b_inds_ref[0, :] = b_inds_ref[0, :] / \
                    self.input_size[1]*inputs_ref.size(3)
                b_inds_ref[1, :] = b_inds_ref[1, :] / \
                    self.input_size[0]*inputs_ref.size(2)
                b_inds_other[0, :] = b_inds_other[0, :] / \
                    self.input_size[1]*inputs_ref.size(3)
                b_inds_other[1, :] = b_inds_other[1, :] / \
                    self.input_size[0]*inputs_ref.size(2)

                min_distance_for_missmatches_scaled = float(
                    self.min_distance_for_missmatches)/self.input_size[1]*inputs_ref.size(3)
            else:
                min_distance_for_missmatches_scaled = self.min_distance_for_missmatches

            b_inds_ref = (b_inds_ref + .5).type(torch.int64)
            b_inds_other = (b_inds_other + .5).type(torch.int64)

            valid_inds = (b_inds_ref[0, :] < inputs_ref.size(3)) & (b_inds_ref[0, :] >= 0) & (b_inds_ref[1, :] < inputs_ref.size(2)) & (b_inds_ref[1, :] >= 0) & (
                b_inds_other[0, :] < inputs_ref.size(3)) & (b_inds_other[0, :] >= 0) & (b_inds_other[1, :] < inputs_ref.size(2)) & (b_inds_other[1, :] >= 0)

            inds_ref_this = b_inds_ref[:, valid_inds]
            inds_other_this = b_inds_other[:, valid_inds]
            weights_this = b_weights[valid_inds, 0]

            n_matches = inds_ref_this.size(1)

            lin_inds_ref = inputs_ref.size(
                2)*inds_ref_this[1, :] + inds_ref_this[0, :]
            lin_inds_other = inputs_ref.size(
                2)*inds_other_this[1, :] + inds_other_this[0, :]

            lin_inds_ref_missmatch = torch.zeros(
                self.n_not_matching*n_matches, dtype=lin_inds_ref.dtype).to(lin_inds_ref.device)
            lin_inds_other_missmatch = torch.zeros(
                self.n_not_matching*n_matches, dtype=lin_inds_other.dtype).to(lin_inds_other.device)
            weights_missmatch = torch.zeros(
                self.n_not_matching*n_matches, dtype=weights_this.dtype).to(weights_this.device)
            # loop through all matches, for each match add a randomized missmatch for which we train the features to be different
            current_ind = 0
            for match_ind in range(n_matches):
                n_added = 0
                n_tries = 0
                while n_added < self.n_not_matching:
                    rand_ind = torch.randint(
                        n_matches, (1,), dtype=torch.int64)
                    distance_between_matches = (
                        (inds_ref_this[:, match_ind]-inds_ref_this[:, rand_ind])**2).sum().type(torch.float32).sqrt()

                    if (distance_between_matches > min_distance_for_missmatches_scaled):
                        # add missmatch
                        lin_inds_ref_missmatch[current_ind] = lin_inds_ref[match_ind]
                        lin_inds_other_missmatch[current_ind] = lin_inds_other[rand_ind]
                        if self.loss_type == 'hingeC' or self.loss_type == 'hingeF':
                            weights_missmatch[current_ind] = 1.0 / \
                                self.n_not_matching
                        else:
                            weights_missmatch[current_ind] = -self.non_match_weight * \
                                distance_between_matches/max_distance/self.n_not_matching

                        n_added += 1
                        current_ind += 1
                        n_tries = 0
                    else:
                        n_tries += 1
                        if n_tries > 50:
                            print("breaking")
                            break

            if self.loss_type == 'L2':
                loss_match = torch.max(torch.sum((inp_ref_flat[b, :, lin_inds_ref]-inp_other_flat[b, :, lin_inds_other])**2, 0), torch.tensor(
                    feat_dist_thresh_match, dtype=torch.float32, device=weights_this.device))
                loss_missmatch = torch.min(torch.sum((inp_ref_flat[b, :, lin_inds_ref_missmatch]-inp_other_flat[b, :, lin_inds_other_missmatch])**2, 0), torch.tensor(
                    feat_dist_thresh_nomatch, dtype=torch.float32, device=weights_this.device))
            elif self.loss_type == 'L1':
                loss_match = torch.max(torch.sum(torch.abs(inp_ref_flat[b, :, lin_inds_ref]-inp_other_flat[b, :, lin_inds_other]), 0), torch.tensor(
                    feat_dist_thresh_match, dtype=torch.float32, device=weights_this.device))
                loss_missmatch = torch.min(torch.sum(torch.abs(inp_ref_flat[b, :, lin_inds_ref_missmatch]-inp_other_flat[b, :, lin_inds_other_missmatch]), 0), torch.tensor(
                    feat_dist_thresh_nomatch, dtype=torch.float32, device=weights_this.device))
            elif self.loss_type == 'hingeC' or self.loss_type == 'hingeF':
                # max(m_p - dt'ds,0)
                loss_match = torch.max(feat_dist_thresh_match - torch.sum(
                    inp_ref_flat[b, :, lin_inds_ref]*inp_other_flat[b, :, lin_inds_other], 0), torch.tensor(0., dtype=torch.float32, device=weights_this.device))
                # max(dt'ds - m_n,0)
                loss_missmatch = torch.max(torch.sum(inp_ref_flat[b, :, lin_inds_ref_missmatch]*inp_other_flat[b, :, lin_inds_other_missmatch],
                                                     0) - feat_dist_thresh_nomatch, torch.tensor(0., dtype=torch.float32, device=weights_this.device))
            elif self.loss_type == 'KL':
                loss_match = F.kl_div(torch.log(
                    inp_other_flat[b, :, lin_inds_other]), inp_ref_flat[b, :, lin_inds_ref])
                # this is later weighted with a negative weight!
                loss_missmatch = F.kl_div(torch.log(
                    inp_other_flat[b, :, lin_inds_other_missmatch]), inp_ref_flat[b, :, lin_inds_ref_missmatch])
            else:
                raise RuntimeError('Specified loss not implemented')

            loss_b = torch.sum(weights_this*loss_match) + \
                torch.sum(weights_missmatch*loss_missmatch)
            if self.size_average:
                loss_b = loss_b/((self.n_not_matching+1.0)*n_matches)
            loss += loss_b

        return loss
