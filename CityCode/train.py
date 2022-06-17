import torch
from tqdm import tqdm
from Utils.ramps import *
from itertools import cycle
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from Base.base_trainer import BaseTrainer
from Utils.sliding_evaluator import SlidingEval
from Utils.metrics import eval_metrics, AverageMeter


class Trainer(BaseTrainer):
    def __init__(self, model, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, wandb_run=None, args=None):
        super(Trainer, self).__init__(model, config, iter_per_epoch, train_logger, args)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.tensor_board = wandb_run if self.args.local_rank <= 0 else None
        self.iter_per_epoch = iter_per_epoch
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1
        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.gamma = config['trainer']['gamma']
        self.evaluator = SlidingEval(model=self.model,
                                     crop_size=800,
                                     stride_rate=2/3,
                                     device="cuda:0" if self.args.local_rank < 0 else
                                     "cuda:{}".format(self.args.local_rank),
                                     val_id=1)


    @torch.no_grad()
    def update_teachers(self, teacher_encoder, teacher_decoder, keep_rate=0.996):
        student_encoder_dict = self.model.module.encoder_s.state_dict()
        student_decoder_dict = self.model.module.decoder_s.state_dict()
        new_teacher_encoder_dict = OrderedDict()
        new_teacher_decoder_dict = OrderedDict()

        for key, value in teacher_encoder.state_dict().items():

            if key in student_encoder_dict.keys():
                new_teacher_encoder_dict[key] = (
                        student_encoder_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student encoder model".format(key))

        for key, value in teacher_decoder.state_dict().items():

            if key in student_decoder_dict.keys():
                new_teacher_decoder_dict[key] = (
                        student_decoder_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student decoder model".format(key))
        teacher_encoder.load_state_dict(new_teacher_encoder_dict, strict=True)
        teacher_decoder.load_state_dict(new_teacher_decoder_dict, strict=True)

    @staticmethod
    def rand_bbox_2(size, n_boxes=1,
                    prop_range=(.0, .5)):
        mask_props = np.random.uniform(prop_range[0], prop_range[1], size=(size[0], n_boxes))
        # mask_props = np.clip(mask_props, .0, 1.)

        mask_shape = (size[3], size[2])
        zero_mask = mask_props == 0.0
        y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(size[0], n_boxes)) * np.log(mask_props))
        x_props = mask_props / y_props
        fac = np.sqrt(1.0 / n_boxes)
        y_props *= fac
        x_props *= fac
        y_props[zero_mask] = 0
        x_props[zero_mask] = 0
        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])
        positions = np.round((np.array(mask_shape) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=2).astype(int)
        bby1 = rectangles[:, :, 0].squeeze().T
        bbx1 = rectangles[:, :, 1].squeeze().T
        bby2 = rectangles[:, :, 2].squeeze().T
        bbx2 = rectangles[:, :, 3].squeeze().T

        return bbx1, bby1, bbx2, bby2

    def cut_mix(self, labeled_image, labeled_mask,
                unlabeled_image=None, unlabeled_mask=None):

        mix_unlabeled_image = unlabeled_image.clone()
        mix_unlabeled_target = unlabeled_mask.clone()
        unsup_boxes = 3
        u_rand_index = torch.randint(low=0, high=unlabeled_image.shape[0],
                                     size=[unsup_boxes, mix_unlabeled_image.shape[0]])

        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox_2(size=unlabeled_image.size(),
                                                          n_boxes=unsup_boxes, prop_range=(0.25, 0.50))

        for i in range(0, mix_unlabeled_image.shape[0]):
            for n in range(0, unsup_boxes):
                mix_unlabeled_image[i, :, u_bbx1[n][i]:u_bbx2[n][i], u_bby1[n][i]:u_bby2[n][i]] = \
                    unlabeled_image[u_rand_index[n][i], :, u_bbx1[n][i]:u_bbx2[n][i], u_bby1[n][i]:u_bby2[n][i]]

                mix_unlabeled_target[i, :, u_bbx1[n][i]:u_bbx2[n][i], u_bby1[n][i]:u_bby2[n][i]] = \
                    unlabeled_mask[u_rand_index[n][i], :, u_bbx1[n][i]:u_bbx2[n][i], u_bby1[n][i]:u_bby2[n][i]]

        del unlabeled_image, unlabeled_mask

        return labeled_image, labeled_mask, mix_unlabeled_image, mix_unlabeled_target

    def predict_with_out_grad(self, image):
        with torch.no_grad():
            predict_target_ul1 = self.model.module.decoder1(self.model.module.encoder1(image),
                                                            data_shape=[image.shape[-2], image.shape[-1]])
            predict_target_ul2 = self.model.module.decoder2(self.model.module.encoder2(image),
                                                            data_shape=[image.shape[-2], image.shape[-1]])
            predict_target_ul1 = torch.nn.functional.interpolate(predict_target_ul1,
                                                                 size=(image.shape[-2], image.shape[-1]),
                                                                 mode='bilinear',
                                                                 align_corners=True)

            predict_target_ul2 = torch.nn.functional.interpolate(predict_target_ul2,
                                                                 size=(image.shape[-2], image.shape[-1]),
                                                                 mode='bilinear',
                                                                 align_corners=True)

            assert predict_target_ul1.shape == predict_target_ul2.shape, "Expect two prediction in same shape,"
        return predict_target_ul1, predict_target_ul2

    # NOTE: the func in here doesn't bring improvements, but stabilize the early stage's training curve.
    def assist_mask_calculate(self, core_predict, assist_predict, topk=1):
        _, index = torch.topk(assist_predict, k=topk, dim=1)
        mask = torch.nn.functional.one_hot(index.squeeze())
        # k!= 1, sum them
        mask = mask.sum(dim=1) if topk > 1 else mask
        if mask.shape[-1] != self.num_classes:
            expand = torch.zeros(
                [mask.shape[0], mask.shape[1], mask.shape[2], self.num_classes - mask.shape[-1]]).cuda()
            mask = torch.cat((mask, expand), dim=3)
        mask = mask.permute(0, 3, 1, 2)
        # get the topk result of the assist map
        assist_predict = torch.mul(assist_predict, mask)

        # fullfill with core predict value for the other entries;
        # as it will be merged based on threshold value
        assist_predict[torch.where(assist_predict == .0)] = core_predict[torch.where(assist_predict == .0)]
        return assist_predict

    def _warm_up(self, epoch, id):
        self.model.train()
        assert id == 1 or id == 2 or id == 3, "Expect ID in 1, 2 or 3"
        dataloader = iter(self.supervised_loader)
        tbar = range(len(self.supervised_loader))

        if self.args.ddp:
            self.supervised_loader.sampler.set_epoch(epoch=epoch-1)

        tbar = tqdm(tbar, ncols=135) if self.args.local_rank <= 0 else tbar
        self._reset_metrics()
        for batch_idx in tbar:
            (input_l_wk, input_l_str, target_l) = next(dataloader)

            input_l = input_l_wk if id == 1 or id == 2 else input_l_str

            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)

            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=None,
                                                         target_ul=None,
                                                         curr_iter=batch_idx, epoch=epoch-1, id=id, warm_up=True)
            if id == 1:
                self.optimizer1.zero_grad()
            elif id == 2:
                self.optimizer2.zero_grad()
            else:
                self.optimizer_s.zero_grad()

            total_loss = total_loss.mean()
            total_loss.backward()
            if id == 1:
                self.optimizer1.step()
            elif id == 2:
                self.optimizer2.step()
            else:
                self.optimizer_s.step()

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, None, sup=True)
            _ = self._log_values(cur_losses)

            del input_l, target_l
            del total_loss, cur_losses, outputs

            if self.args.local_rank <= 0:
                tbar.set_description('ID {} Warm ({}) | Ls {:.2f} |'.format(id, epoch, self.loss_sup.average))

        return

    def _train_epoch(self, epoch, id):
        assert id == 1 or id == 2, "Expect ID in 1 or 2"
        self.model.module.freeze_teachers_parameters()
        self.model.train()
        if self.args.ddp:
            self.supervised_loader.sampler.set_epoch(epoch=epoch - 1)
            self.unsupervised_loader.sampler.set_epoch(epoch=epoch - 1)
        dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        tbar = range(len(self.unsupervised_loader))

        tbar = tqdm(tbar, ncols=135, leave=True) if self.args.local_rank <= 0 else tbar
        self._reset_metrics()
        for batch_idx in tbar:
            if self.args.local_rank <= 0:
                self.tensor_board.step_forward(len(self.unsupervised_loader) * (epoch - 1) + batch_idx)
            if self.mode == "semi":
                (_, input_l, target_l), (input_ul_wk, input_ul_str, target_ul) = next(dataloader)
                input_ul_wk, input_ul_str, target_ul = input_ul_wk.cuda(non_blocking=True), \
                                                       input_ul_str.cuda(non_blocking=True), \
                                                       target_ul.cuda(non_blocking=True)
            else:
                (_, input_l, target_l) = next(dataloader)
                input_ul_wk, input_ul_str, target_ul = None, None, None

            # strong aug for all the supervised images
            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)

            # predicted unlabeled data
            if self.mode == "semi":
                t1_prob, t2_prob = self.predict_with_out_grad(input_ul_wk)
                if id == 1:
                    t2_prob = self.assist_mask_calculate(core_predict=t1_prob,
                                                         assist_predict=t2_prob,
                                                         topk=7)

                else:
                    t1_prob = self.assist_mask_calculate(core_predict=t2_prob,
                                                         assist_predict=t1_prob,
                                                         topk=7)

                predict_target_ul = self.gamma * t1_prob + (1 - self.gamma) * t2_prob
            else:
                predict_target_ul = None

            origin_predict = predict_target_ul.detach().clone()

            if batch_idx == 0 or batch_idx == int(len(self.unsupervised_loader) / 2):
                if self.args.local_rank <= 0:
                    self.tensor_board.update_wandb_city_image(images=input_ul_wk,
                                                              ground_truth=target_ul,
                                                              teacher_prediction=predict_target_ul)

            input_l, target_l, input_ul_str, predict_target_ul = self.cut_mix(input_l, target_l,
                                                                              input_ul_str,
                                                                              predict_target_ul)

            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l,
                                                         x_ul=input_ul_str,
                                                         target_ul=predict_target_ul,
                                                         curr_iter=batch_idx, epoch=epoch - 1, id=id,
                                                         semi_p_th=self.args.semi_p_th,
                                                         semi_n_th=self.args.semi_n_th)
            total_loss = total_loss.mean()

            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()
            outputs['unsup_pred'] = origin_predict
            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul,
                                  sup=True if self.model.module.mode == "supervised" else False)

            _ = self._log_values(cur_losses)

            if self.args.local_rank <= 0:
                if batch_idx == 0 or batch_idx == int(len(self.unsupervised_loader) / 2):
                    self.tensor_board.update_table(cur_losses['pass_rate']['entire_prob_boundary'],
                                                   axis_name={"x": "boundary", "y": "rate"},
                                                   title="pass_in_each_boundary")

                    self.tensor_board.update_table(cur_losses['pass_rate']['max_prob_boundary'],
                                                   axis_name={"x": "boundary", "y": "rate"},
                                                   title="max_prob_in_each_boundary")

                if batch_idx % self.log_step == 0:
                    for i, opt_group in enumerate(self.optimizer_s.param_groups[:2]):
                        self.tensor_board.upload_single_info({f"learning_rate_{i}": opt_group['lr']})
                    self.tensor_board.upload_single_info({"ramp_up": self.model.module.unsup_loss_w.current_rampup})

                tbar.set_description('ID {} T ({}) | Ls {:.3f} Lu {:.3f} Lw {:.3f} m1 {:.3f} m2 {:.3f}|'.format(
                    id, epoch, self.loss_sup.average, self.loss_unsup.average, self.loss_weakly.average,
                    self.mIoU_l, self.mIoU_ul))

            if self.args.ddp:
                dist.barrier()

            del input_l, target_l, input_ul_wk, input_ul_str, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler_s.step(epoch=epoch - 1)

            with torch.no_grad():
                if id == 1:
                    self.update_teachers(teacher_encoder=self.model.module.encoder1,
                                         teacher_decoder=self.model.module.decoder1)
                else:
                    self.update_teachers(teacher_encoder=self.model.module.encoder2,
                                         teacher_decoder=self.model.module.decoder2)
                if self.args.ddp:
                    dist.barrier()

        return

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        stride = int(np.ceil(len(self.val_loader.dataset) / self.args.gpus))
        current_rank = max(0, self.args.local_rank)
        e_record = min((current_rank + 1) * stride, len(self.val_loader.dataset))
        shred_list = list(range(current_rank * stride, e_record))
        total_inter, total_union = torch.tensor(0), torch.tensor(0)
        total_correct, total_label = torch.tensor(0), torch.tensor(0)
        tbar = tqdm(shred_list, ncols=130, leave=True) if self.args.local_rank <= 0 else shred_list
        with torch.no_grad():
            for batch_idx in tbar:
                data, target = self.val_loader.dataset[batch_idx]
                target, data = target.unsqueeze(0).cuda(non_blocking=True), data.unsqueeze(0).cuda(non_blocking=True)

                output = self.evaluator(img=data)
                output = torch.tensor(output, dtype=torch.float).cuda(non_blocking=True).unsqueeze(0)
                target = target.cuda(non_blocking=True)

                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

            if self.args.gpus > 1:
                total_inter = torch.tensor(total_inter, device=self.args.local_rank)
                total_union = torch.tensor(total_union, device=self.args.local_rank)
                total_correct = torch.tensor(total_correct, device=self.args.local_rank)
                total_label = torch.tensor(total_label, device=self.args.local_rank)
                dist.all_reduce(total_inter, dist.ReduceOp.SUM)
                dist.all_reduce(total_union, dist.ReduceOp.SUM)
                dist.all_reduce(total_correct, dist.ReduceOp.SUM)
                dist.all_reduce(total_label, dist.ReduceOp.SUM)

            # PRINT INFO
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean().item()
            pixAcc = pixAcc.item()
            seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 4),
                           "Mean_IoU": np.round(mIoU, 4),
                           "Class_IoU": dict(zip(range(19),
                                                 np.round(IoU.cpu().numpy(), 4)))}

        if self.args.local_rank <= 0:
            print('EVAL ID ({}) ({}) | PixelAcc: {:.4f}, Mean IoU: {:.4f} |'.format(
                "Model 1",
                epoch,
                pixAcc,
                mIoU))
            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }
            valid_dict = {}
            for k, v in list(seg_metrics.items())[:-1]:
                valid_dict[f'valid_{k}'] = v
            self.tensor_board.upload_wandb_info(valid_dict)

            return log

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    def _update_losses(self, cur_losses):
        if "loss_sup" in cur_losses.keys():
            self.loss_sup.update(cur_losses['loss_sup'].mean().item())
        if "loss_unsup" in cur_losses.keys():
            self.loss_unsup.update(cur_losses['loss_unsup'].mean().item())
        if "loss_weakly" in cur_losses.keys():
            self.loss_weakly.update(cur_losses['loss_weakly'].mean().item())
        if "pair_wise" in cur_losses.keys():
            self.pair_wise.update(cur_losses['pair_wise'].mean().item())

    def _compute_metrics(self, outputs, target_l, target_ul, sup=False):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)
        self._update_seg_metrics(*seg_metrics_l, True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if sup:
            return

        if self.mode == 'semi':
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)
            self._update_seg_metrics(*seg_metrics_ul, False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average

        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average

        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l

        if self.args.local_rank <= 0:
            self.tensor_board.upload_single_info({'loss_sup': self.loss_sup.average})
            self.tensor_board.upload_single_info({'mIoU_labeled': self.mIoU_l})
            self.tensor_board.upload_single_info({'pixel_acc_labeled': self.pixel_acc_l})

            if self.mode == 'semi':
                logs['mIoU_unlabeled'] = self.mIoU_ul
                logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
                self.tensor_board.upload_single_info({'loss_unsup': self.loss_unsup.average})
                self.tensor_board.upload_single_info({'mIoU_unlabeled':  self.mIoU_ul})
                self.tensor_board.upload_single_info({'pixel_acc_unlabeled': self.pixel_acc_ul})

        return logs
