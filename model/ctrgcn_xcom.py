import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ctrgcn import Model as CTRGCNModel
from model.xcom_branch import XCoMTemporalBranch


class TemporalGatedFusion(nn.Module):
    """Fuse temporally aligned CTR-GCN and XCoM features."""

    def __init__(self, channels=256, residual_scale_init=0.0):
        super(TemporalGatedFusion, self).__init__()
        self.channels = int(channels)
        self.gate = nn.Conv1d(self.channels * 2, self.channels, kernel_size=1)
        nn.init.kaiming_normal_(self.gate.weight, mode='fan_out')
        nn.init.constant_(self.gate.bias, -2.0)
        self.residual_scale = nn.Parameter(torch.tensor(
            float(residual_scale_init), dtype=torch.float32
        ))

    def forward(self, ctr_features, xcom_features):
        if ctr_features.dim() != 3 or xcom_features.dim() != 3:
            raise ValueError(
                'ctr_features and xcom_features must have shape [P,C,T].'
            )
        if ctr_features.size(0) != xcom_features.size(0):
            raise ValueError('CTR-GCN and XCoM person dimensions do not match.')
        if ctr_features.size(1) != self.channels:
            raise ValueError('Unexpected CTR-GCN feature channel dimension.')
        if xcom_features.size(1) != self.channels:
            raise ValueError('Unexpected XCoM feature channel dimension.')

        target_length = ctr_features.size(-1)
        if xcom_features.size(-1) != target_length:
            xcom_features = F.adaptive_avg_pool1d(xcom_features, target_length)

        concatenated = torch.cat((ctr_features, xcom_features), dim=1)
        gate = torch.sigmoid(self.gate(concatenated))
        fused = (
            ctr_features
            + self.residual_scale * gate * xcom_features
        )
        return {
            'fused_features': fused,  # P,256,T
            'gate': gate,  # P,256,T
            'ctr_features': ctr_features,  # P,256,T
            'xcom_features': xcom_features,  # P,256,T
        }


class Model(CTRGCNModel):
    """CTR-GCN and XCoM dual-branch action-recognition model.

    The inherited CTR-GCN parameter names remain unchanged, allowing an
    existing CTR-GCN checkpoint to initialize data_bn, l1-l10, and fc. New
    parameters are stored under xcom_branch.* and fusion.*.
    """

    def __init__(
        self,
        num_class=60,
        num_point=17,
        num_person=2,
        graph=None,
        graph_args=None,
        in_channels=5,
        drop_out=0,
        adaptive=True,
        frame_interval=1.0,
        xcom_branch_args=None,
        fusion_args=None,
        freeze_backbone_epochs=10,
        partial_backbone_blocks=('l8', 'l9', 'l10'),
        full_unfreeze_epoch=-1,
        freeze_data_bn=True,
        backbone_lr_scale=0.1,
    ):
        graph_args = dict(graph_args or {})
        if int(num_point) != 17:
            raise ValueError('The XCoM branch requires 17 COCO joints.')
        if int(in_channels) < 4:
            raise ValueError(
                'The dual-branch model requires abs_x, abs_y, rel_x, rel_y.'
            )
        if float(frame_interval) <= 0:
            raise ValueError('frame_interval must be positive.')

        super(Model, self).__init__(
            num_class=int(num_class),
            num_point=int(num_point),
            num_person=int(num_person),
            graph=graph,
            graph_args=graph_args,
            in_channels=int(in_channels),
            drop_out=drop_out,
            adaptive=adaptive,
        )
        self.input_channels = int(in_channels)
        self.num_person = int(num_person)
        self.frame_interval = float(frame_interval)
        self.feature_channels = 256
        self.freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))
        self.partial_backbone_blocks = tuple(partial_backbone_blocks)
        self.full_unfreeze_epoch = int(full_unfreeze_epoch)
        self.freeze_data_bn = bool(freeze_data_bn)
        self.backbone_lr_scale = float(backbone_lr_scale)
        if not 0 < self.backbone_lr_scale <= 1:
            raise ValueError('backbone_lr_scale must lie in (0,1].')
        valid_blocks = {'l{}'.format(index) for index in range(1, 11)}
        unknown_blocks = set(self.partial_backbone_blocks).difference(valid_blocks)
        if unknown_blocks:
            raise ValueError(
                'Unknown partial_backbone_blocks: {}'.format(
                    sorted(unknown_blocks)
                )
            )

        branch_args = dict(xcom_branch_args or {})
        branch_output_channels = int(
            branch_args.setdefault('output_channels', self.feature_channels)
        )
        if branch_output_channels != self.feature_channels:
            raise ValueError('XCoM output_channels must be 256 for CTR-GCN fusion.')
        self.xcom_branch = XCoMTemporalBranch(**branch_args)

        fusion_config = dict(fusion_args or {})
        fusion_channels = int(
            fusion_config.setdefault('channels', self.feature_channels)
        )
        if fusion_channels != self.feature_channels:
            raise ValueError('Fusion channels must be 256.')
        self.fusion = TemporalGatedFusion(**fusion_config)
        self._training_epoch = 0
        self._training_stage = 'stage1'
        self._schedule_state = None
        self._apply_training_schedule()

    @staticmethod
    def _set_module_trainable(module, trainable, training_mode):
        for parameter in module.parameters():
            parameter.requires_grad_(bool(trainable))
        if training_mode and trainable:
            module.train()
        else:
            module.eval()

    def _schedule_phase(self):
        if self._training_stage != 'stage1':
            return 'partial_unfreeze'
        if self._training_epoch < self.freeze_backbone_epochs:
            return 'xcom_warmup'
        if (
            self.full_unfreeze_epoch >= 0
            and self._training_epoch >= self.full_unfreeze_epoch
        ):
            return 'full_unfreeze'
        return 'partial_unfreeze'

    def _apply_training_schedule(self):
        phase = self._schedule_phase()
        training_mode = bool(self.training)

        data_bn_trainable = phase == 'full_unfreeze' and not self.freeze_data_bn
        self._set_module_trainable(
            self.data_bn, data_bn_trainable, training_mode
        )
        for block_index in range(1, 11):
            block_name = 'l{}'.format(block_index)
            if phase == 'full_unfreeze':
                trainable = True
            elif phase == 'partial_unfreeze':
                trainable = block_name in self.partial_backbone_blocks
            else:
                trainable = False
            self._set_module_trainable(
                getattr(self, block_name), trainable, training_mode
            )

        self._set_module_trainable(self.xcom_branch, True, training_mode)
        self._set_module_trainable(self.fusion, True, training_mode)
        self._set_module_trainable(self.fc, True, training_mode)
        return phase

    def set_training_stage(self, epoch, stage_name='stage1'):
        self._training_epoch = int(epoch)
        self._training_stage = str(stage_name)
        previous_state = self._schedule_state
        phase = self._apply_training_schedule()
        self._schedule_state = (
            self._training_stage,
            phase,
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            'stage': self._training_stage,
            'phase': phase,
            'epoch': self._training_epoch,
            'trainable_parameters': int(trainable),
            'total_parameters': int(total),
            'changed': previous_state != self._schedule_state,
        }

    def train(self, mode=True):
        super(Model, self).train(mode)
        if hasattr(self, '_training_epoch'):
            self._apply_training_schedule()
        return self

    def build_optimizer_param_groups(self, base_lr):
        base_lr = float(base_lr)
        backbone_prefixes = tuple(
            ['data_bn.'] + ['l{}.'.format(index) for index in range(1, 11)]
        )
        backbone_parameters = []
        new_parameters = []
        for name, parameter in self.named_parameters():
            if name.startswith(backbone_prefixes):
                backbone_parameters.append(parameter)
            else:
                new_parameters.append(parameter)
        return [
            {
                'params': backbone_parameters,
                'lr': base_lr * self.backbone_lr_scale,
                'lr_scale': self.backbone_lr_scale,
                'group_name': 'ctr_backbone',
            },
            {
                'params': new_parameters,
                'lr': base_lr,
                'lr_scale': 1.0,
                'group_name': 'xcom_fusion_fc',
            },
        ]

    def _prepare_input(self, x):
        if x.dim() == 3:
            n, time_steps, flattened = x.shape
            one_person_features = self.num_point * self.input_channels
            all_person_features = self.num_person * one_person_features
            if flattened == one_person_features:
                x = x.view(
                    n, time_steps, self.num_point, self.input_channels
                ).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
                if self.num_person > 1:
                    padding = x.new_zeros(
                        n,
                        self.input_channels,
                        time_steps,
                        self.num_point,
                        self.num_person - 1,
                    )
                    x = torch.cat((x, padding), dim=-1)
            elif flattened == all_person_features:
                x = x.view(
                    n,
                    time_steps,
                    self.num_person,
                    self.num_point,
                    self.input_channels,
                ).permute(0, 4, 1, 3, 2).contiguous()
            else:
                raise ValueError('Flattened skeleton feature dimension is invalid.')

        if x.dim() != 5:
            raise ValueError('Input must have shape [N,C,T,V,M] or [N,T,MVC].')
        n, channels, time_steps, joints, persons = x.shape
        if channels != self.input_channels:
            raise ValueError(
                'Input channels {} do not match configured in_channels {}.'.format(
                    channels, self.input_channels
                )
            )
        if joints != self.num_point:
            raise ValueError(
                'Input joints {} do not match configured num_point {}.'.format(
                    joints, self.num_point
                )
            )
        if persons > self.num_person:
            raise ValueError(
                'Input persons {} exceed configured num_person {}.'.format(
                    persons, self.num_person
                )
            )
        if persons < self.num_person:
            padding = x.new_zeros(
                n,
                channels,
                time_steps,
                joints,
                self.num_person - persons,
            )
            x = torch.cat((x, padding), dim=-1)
        return x

    def _forward_ctr_branch(self, skeleton):
        n, channels, time_steps, joints, persons = skeleton.shape
        x = skeleton.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n, persons * joints * channels, time_steps)
        x = self.data_bn(x)
        x = x.view(n, persons, joints, channels, time_steps)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * persons, channels, time_steps, joints)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        return self.l10(x)  # P,256,T_ctr,17

    def forward_features(self, x, frame_interval=None):
        skeleton = self._prepare_input(x)
        n, _channels, _time_steps, _joints, persons = skeleton.shape
        interval = self.frame_interval if frame_interval is None else float(frame_interval)
        if interval <= 0:
            raise ValueError('frame_interval must be positive.')

        ctr_spatiotemporal = self._forward_ctr_branch(skeleton)
        ctr_temporal = ctr_spatiotemporal.mean(dim=-1)
        xcom_result = self.xcom_branch(
            skeleton,
            target_length=ctr_temporal.size(-1),
            frame_interval=interval,
        )
        fusion_result = self.fusion(
            ctr_temporal,
            xcom_result['projected_features_flat'],
        )
        fused_temporal = fusion_result['fused_features']
        person_features = fused_temporal.mean(dim=-1)
        person_features = person_features.view(
            n, persons, self.feature_channels
        )
        sample_features = person_features.mean(dim=1)

        return {
            'sample_features': sample_features,  # N,256
            'person_features': person_features,  # N,M,256
            'ctr_spatiotemporal': ctr_spatiotemporal,  # P,256,T_ctr,17
            'fusion': fusion_result,
            'xcom': xcom_result,
        }

    def forward(self, x, return_auxiliary=False, frame_interval=None):
        features = self.forward_features(x, frame_interval=frame_interval)
        classifier_input = self.drop_out(features['sample_features'])
        logits = self.fc(classifier_input)
        if not return_auxiliary:
            return logits

        return {
            'logits': logits,
            'classifier_input': classifier_input,
            **features,
        }
