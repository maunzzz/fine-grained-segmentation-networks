import os
import numpy as np
import torch
import pickle
from PIL import Image
from utils.misc import AverageMeter, evaluate_incremental, freeze_bn
from utils.segmentor import Segmentor, FeatureExtractor
import torchvision.transforms.functional as F
from clustering.cluster_tools import assign_cluster_ids_to_correspondence_points
from clustering.clustering import preprocess_features


class Validator():
    def __init__(self, data_loader, n_classes=19, save_snapshot=False, extra_name_str=''):
        self.data_loader = data_loader
        self.n_classes = n_classes
        self.save_snapshot = save_snapshot
        self.extra_name_str = extra_name_str

    def run(self, net, optimizer, args, curr_iter, save_dir, f_handle, writer=None):
        # the following code is written assuming that batch size is 1
        net.eval()
        segmentor = Segmentor(net, self.n_classes, colorize_fcn=None, n_slices_per_pass=10)

        confmat = np.zeros((self.n_classes, self.n_classes))
        for vi, data in enumerate(self.data_loader):
            img_slices, gt, slices_info = data
            gt.squeeze_(0)
            prediction_tmp = segmentor.run_on_slices(img_slices.squeeze_(0), slices_info.squeeze_(0))

            if prediction_tmp.shape != gt.size():
                prediction_tmp = Image.fromarray(prediction_tmp.astype(np.uint8)).convert('P')
                prediction_tmp = F.resize(prediction_tmp, gt.size(), interpolation=Image.NEAREST)

            acc, acc_cls, mean_iu, fwavacc, confmat, _ = evaluate_incremental(
                confmat, np.asarray(prediction_tmp), gt.numpy(), self.n_classes)

            str2write = 'validating: %d / %d' % (vi + 1, len(self.data_loader))
            print(str2write)
            # f_handle.write(str2write + "\n")

        # Store confusion matrix
        confmatdir = os.path.join(save_dir, 'confmat')
        os.makedirs(confmatdir, exist_ok=True)
        with open(os.path.join(confmatdir, self.extra_name_str + str(curr_iter) + '_confmat.pkl'), 'wb') as confmat_file:
            pickle.dump(confmat, confmat_file)

        if self.save_snapshot:
            snapshot_name = 'iter_%d_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
                curr_iter, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr'])
            torch.save(net.state_dict(), os.path.join(
                save_dir, snapshot_name + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(
                save_dir, 'opt_' + snapshot_name + '.pth'))

            if args['best_record']['mean_iu'] < mean_iu:
                args['best_record']['iter'] = curr_iter
                args['best_record']['acc'] = acc
                args['best_record']['acc_cls'] = acc_cls
                args['best_record']['mean_iu'] = mean_iu
                args['best_record']['fwavacc'] = fwavacc
                args['best_record']['snapshot'] = snapshot_name
                open(os.path.join(save_dir, 'bestval.txt'), 'w').write(
                    str(args['best_record']) + '\n\n')

            str2write = '%s best record: [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (self.extra_name_str,
                                                                                                        args['best_record']['acc'], args['best_record']['acc_cls'], args['best_record']['mean_iu'], args['best_record']['fwavacc'])

            print(str2write)
            f_handle.write(str2write + "\n")

        str2write = '%s [iter %d], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (self.extra_name_str,
                                                                                                  curr_iter, acc, acc_cls, mean_iu, fwavacc)
        print(str2write)
        f_handle.write(str2write + "\n")

        if writer is not None:
            writer.add_scalar(self.extra_name_str + ': acc', acc, curr_iter)
            writer.add_scalar(self.extra_name_str +
                              ': acc_cls', acc_cls, curr_iter)
            writer.add_scalar(self.extra_name_str +
                              ': mean_iu', mean_iu, curr_iter)
            writer.add_scalar(self.extra_name_str +
                              ': fwavacc', fwavacc, curr_iter)

        net.train()
        if 'freeze_bn' not in args or args['freeze_bn']:
            freeze_bn(net)

        return mean_iu


class CorrValidator():
    def __init__(self, data_loader, loss_fct_feat, loss_fct_out, loss_fct_cluster, save_snapshot=False, extra_name_str=''):
        self.data_loader = data_loader
        self.loss_fct_feat = loss_fct_feat
        self.loss_fct_out = loss_fct_out
        self.loss_fct_cluster = loss_fct_cluster
        self.save_snapshot = save_snapshot
        self.extra_name_str = extra_name_str

    def run(self, net, net_for_clustering, cluster_info, optimizer, args, curr_iter, save_dir, f_handle, writer=None):
        # the following code is written assuming that batch size is 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net.eval()
        previous_setting_output_all = net.output_all
        net.output_all = True
        val_loss_feat = AverageMeter()
        val_loss_out = AverageMeter()
        val_loss_cluster = AverageMeter()
        for vi, data in enumerate(self.data_loader):
            img_ref, img_other, pts_ref, pts_other, weights = data

            img_ref = img_ref.to(device)
            img_other = img_other.to(device)
            pts_ref = [p.to(device) for p in pts_ref]
            pts_other = [p.to(device) for p in pts_other]
            weights = [w.to(device) for w in weights]

            with torch.no_grad():
                feats_ref, out_ref = net(img_ref)
                feats_other, out_other = net(img_other)
                features = net_for_clustering(img_ref)

            loss_feat = self.loss_fct_feat(
                feats_ref, feats_other, pts_ref, pts_other, weights)
            loss_out = self.loss_fct_out(
                out_ref, out_other, pts_ref, pts_other, weights)

            cluster_labels = assign_cluster_ids_to_correspondence_points(
                features, pts_ref, cluster_info, inds_other=pts_other, orig_im_size=self.loss_fct_cluster.input_size)
            loss_cluster, _ = self.loss_fct_cluster(
                out_ref, out_other, feats_ref, pts_ref, pts_other, cluster_labels=cluster_labels)

            val_loss_feat.update(loss_feat.item())
            val_loss_out.update(loss_out.item())
            val_loss_cluster.update(loss_cluster.item())

            if (vi % 100) == 0:
                str2write = 'validating: %d / %d' % (
                    vi + 1, len(self.data_loader))
                print(str2write)

        net.output_all = previous_setting_output_all
        net.train()
        if 'freeze_bn' not in args or args['freeze_bn']:
            freeze_bn(net)

        if self.save_snapshot:
            snapshot_name = 'iter_%d_lossfeat_%.5f_lossout_%.5f_lr_%.10f' % (
                curr_iter, val_loss_feat.avg, val_loss_out.avg, optimizer.param_groups[1]['lr'])
            torch.save(net.state_dict(), os.path.join(
                save_dir, snapshot_name + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(
                save_dir, 'opt_' + snapshot_name + '.pth'))

            if args['best_record']['val_loss_cluster'] > val_loss_cluster.avg:
                args['best_record']['val_loss_cluster'] = val_loss_cluster.avg
                args['best_record']['val_loss_feat'] = val_loss_feat.avg
                args['best_record']['val_loss_out'] = val_loss_out.avg
                args['best_record']['iter'] = curr_iter
                args['best_record']['snapshot'] = snapshot_name
                open(os.path.join(save_dir, 'bestval.txt'), 'w').write(
                    str(args['best_record']) + '\n\n')

            str2write = '%s best record: [val loss feat %.5f], [val loss out %.5f], [val loss cluster %.5f]' % (
                self.extra_name_str, args['best_record']['val_loss_feat'], args['best_record']['val_loss_out'], args['best_record']['val_loss_cluster'])

            print(str2write)
            f_handle.write(str2write + "\n")

        str2write = '%s [iter %d], [val loss feat %.5f], [val loss out %.5f], [val loss cluster %.5f]' % (
            self.extra_name_str, curr_iter, val_loss_feat.avg, val_loss_out.avg, val_loss_cluster.avg)
        print(str2write)
        f_handle.write(str2write + "\n")

        if writer is not None:
            writer.add_scalar(self.extra_name_str +
                              ': val_loss_feat', val_loss_feat.avg, curr_iter)
            writer.add_scalar(self.extra_name_str +
                              ': val_loss_out', val_loss_out.avg, curr_iter)
            writer.add_scalar(
                self.extra_name_str + ': val_loss_cluster', val_loss_cluster.avg, curr_iter)

        return val_loss_feat.avg


class ClusterValidator():
    def __init__(self, data_loader, n_clusters, save_snapshot=False, extra_name_str=''):
        self.data_loader = data_loader
        self.save_snapshot = save_snapshot
        self.extra_name_str = extra_name_str
        self.n_clusters = n_clusters

    def run(self, net, net_for_clustering, cluster_info, optimizer, args, curr_iter, save_dir, f_handle, writer=None):
        # the following code is written assuming that batch size is 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net.eval()
        net_for_clustering.eval()
        ps_output_features1 = net_for_clustering.output_features
        ps_output_features2 = net.output_features
        net_for_clustering.output_features = True
        net.output_features = False

        feat_ext_for_cluster = FeatureExtractor(net_for_clustering, n_slices_per_pass=5)
        feat_ext = FeatureExtractor(net, n_slices_per_pass=5)

        confmat = np.zeros((self.n_clusters, self.n_clusters))
        for vi, data in enumerate(self.data_loader):
            img_slices, _, slices_info = data

            output = feat_ext_for_cluster.run_on_slices(img_slices, slices_info)
            seg_out = feat_ext.run_on_slices(img_slices, slices_info)
            seg = np.argmax(seg_out, 0)

            output_flat = output.reshape((output.shape[0], -1))
            out_f2, _ = preprocess_features(np.swapaxes(output_flat, 0, 1), pca_info=cluster_info[1])
            cluster_labels = cluster_info[0].assign(out_f2)
            cluster_image = cluster_labels.reshape((output.shape[1], output.shape[2]))

            acc, acc_cls, mean_iu, fwavacc, confmat, _ = evaluate_incremental(
                confmat, seg, cluster_image, self.n_clusters)

            if (vi % 100) == 0:
                str2write = 'validating: %d / %d' % (
                    vi + 1, len(self.data_loader))
                print(str2write)

        net_for_clustering.output_features = ps_output_features1
        net.output_features = ps_output_features2
        net.train()
        if 'freeze_bn' not in args or args['freeze_bn']:
            freeze_bn(net)

        if self.save_snapshot:
            snapshot_name = 'iter_%d_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
                curr_iter, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr'])
            torch.save(net.state_dict(), os.path.join(save_dir, snapshot_name + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'opt_' + snapshot_name + '.pth'))

            if args['best_record']['mean_iu'] < mean_iu:
                args['best_record']['iter'] = curr_iter
                args['best_record']['acc'] = acc
                args['best_record']['acc_cls'] = acc_cls
                args['best_record']['mean_iu'] = mean_iu
                args['best_record']['fwavacc'] = fwavacc
                args['best_record']['snapshot'] = snapshot_name
                open(os.path.join(save_dir, 'bestval.txt'), 'w').write(
                    str(args['best_record']) + '\n\n')

            str2write = '%s best record: [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (self.extra_name_str,
                                                                                                        args['best_record']['acc'], args['best_record']['acc_cls'], args['best_record']['mean_iu'], args['best_record']['fwavacc'])

            print(str2write)
            f_handle.write(str2write + "\n")

            str2write = '%s [iter %d], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (self.extra_name_str,
                                                                                                      curr_iter, acc, acc_cls, mean_iu, fwavacc)
            print(str2write)
            f_handle.write(str2write + "\n")

            if writer is not None:
                writer.add_scalar(self.extra_name_str + ': acc', acc, curr_iter)
                writer.add_scalar(self.extra_name_str +
                                  ': acc_cls', acc_cls, curr_iter)
                writer.add_scalar(self.extra_name_str +
                                  ': mean_iu', mean_iu, curr_iter)
                writer.add_scalar(self.extra_name_str +
                                  ': fwavacc', fwavacc, curr_iter)

        return mean_iu
