import math

import torch
import torch.nn as nn
import torch.nn.functional as F


COCO17_SCALE_LINKS = (
    (5, 6), (11, 12),
    (5, 11), (6, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)

# Sex-averaged de Leva mass fractions, collapsed to ten COCO17 parts.
COCO17_MASS_PRIOR = (
    0.06810,  # head
    0.43020,  # trunk
    0.02630, 0.02630,  # upper arms
    0.02085, 0.02085,  # forearms and hands
    0.14470, 0.14470,  # thighs
    0.05900, 0.05900,  # shanks and feet
)

# Tie left and right mass corrections to preserve body symmetry.
MASS_GROUP_INDEX = (0, 1, 2, 2, 3, 3, 4, 4, 5, 5)

# Proximal-to-distal center-of-mass ratios for six symmetric segment groups.
KAPPA_GROUP_PRIOR = (
    0.5000,  # head and neck: mid-shoulder to head landmark center
    0.5000,  # trunk: mid-hip to mid-shoulder
    0.5763,  # upper arm: shoulder to elbow
    0.4567,  # forearm: elbow to wrist
    0.3854,  # thigh: hip to knee
    0.4438,  # shank: knee to ankle
)

# Expand the six groups to the same ten body segments used by mass weights.
KAPPA_GROUP_INDEX = MASS_GROUP_INDEX
KAPPA_PRIOR = tuple(
    KAPPA_GROUP_PRIOR[group_index]
    for group_index in KAPPA_GROUP_INDEX
)


def _inverse_sigmoid(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _init_conv(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def _init_linear(linear, final=False):
    if final:
        nn.init.normal_(linear.weight, mean=0.0, std=1e-3)
    else:
        nn.init.kaiming_normal_(linear.weight, mode='fan_out')
    if linear.bias is not None:
        nn.init.constant_(linear.bias, 0)


def _masked_mean_time_last(features, valid_frames, eps):
    """Pool [P,T,C] features to [P,C] over valid frames."""
    weights = valid_frames.to(features.dtype).unsqueeze(-1)
    numerator = (features * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp(min=eps)
    return numerator / denominator


def _masked_mean_time_first(features, valid_frames, eps):
    """Pool [P,C,T] features to [P,C] over valid frames."""
    weights = valid_frames.to(features.dtype).unsqueeze(1)
    numerator = (features * weights).sum(dim=2)
    denominator = weights.sum(dim=2).clamp(min=eps)
    return numerator / denominator


class SamePadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, causal=False):
        super(SamePadConv1d, self).__init__()
        self.causal = bool(causal)
        self.total_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
        )
        _init_conv(self.conv)

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.total_padding, 0))
        else:
            left = self.total_padding // 2
            right = self.total_padding - left
            x = F.pad(x, (left, right))
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, dropout=0.1, causal=False):
        super(TemporalResidualBlock, self).__init__()
        self.conv1 = SamePadConv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation, causal=causal
        )
        self.bn1 = nn.BatchNorm1d(int(out_channels))
        self.conv2 = SamePadConv1d(
            out_channels, out_channels, kernel_size=3, dilation=dilation, causal=causal
        )
        self.bn2 = nn.BatchNorm1d(int(out_channels))
        self.dropout = nn.Dropout(float(dropout))
        if int(in_channels) == int(out_channels):
            self.residual = None
        else:
            self.residual = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
            _init_conv(self.residual)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        y = self.dropout(F.relu(self.bn1(self.conv1(x)), inplace=True))
        y = self.bn2(self.conv2(y))
        return F.relu(y + residual, inplace=True)


class LocalPoseEncoder(nn.Module):
    """Encode every flattened COCO17 relative pose: 34 -> 64 -> 64."""

    def __init__(self, input_channels=34, hidden_channels=64, output_channels=64):
        super(LocalPoseEncoder, self).__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.fc1 = nn.Linear(self.input_channels, int(hidden_channels))
        self.bn1 = nn.BatchNorm1d(int(hidden_channels))
        self.fc2 = nn.Linear(int(hidden_channels), self.output_channels)
        self.bn2 = nn.BatchNorm1d(self.output_channels)
        _init_linear(self.fc1)
        _init_linear(self.fc2)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, relative_pose, valid_frames):
        if relative_pose.dim() != 3 or relative_pose.size(-1) != self.input_channels:
            raise ValueError('relative_pose must have shape [P,T,34].')
        persons, time_steps, _ = relative_pose.shape
        x = relative_pose.reshape(persons * time_steps, self.input_channels)
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = x.view(persons, time_steps, self.output_channels)
        return x * valid_frames.to(x.dtype).unsqueeze(-1)


class GlobalPoseEncoder(nn.Module):
    """Encode a full local-feature sequence: 64 -> 128 -> 128."""

    def __init__(
        self,
        input_channels=64,
        output_channels=128,
        dilations=(1, 2),
        dropout=0.1,
        causal=False,
        eps=1e-6,
    ):
        super(GlobalPoseEncoder, self).__init__()
        if len(dilations) != 2:
            raise ValueError('GlobalPoseEncoder requires exactly two TCN dilations.')
        self.eps = float(eps)
        self.block1 = TemporalResidualBlock(
            int(input_channels),
            int(output_channels),
            dilation=int(dilations[0]),
            dropout=float(dropout),
            causal=bool(causal),
        )
        self.block2 = TemporalResidualBlock(
            int(output_channels),
            int(output_channels),
            dilation=int(dilations[1]),
            dropout=float(dropout),
            causal=bool(causal),
        )

    def forward(self, local_features, valid_frames):
        if local_features.dim() != 3:
            raise ValueError('local_features must have shape [P,T,C].')
        temporal = local_features.permute(0, 2, 1).contiguous()
        temporal = self.block2(self.block1(temporal))
        global_features = _masked_mean_time_first(temporal, valid_frames, self.eps)
        return temporal, global_features


class GlobalLocalParameterEstimator(nn.Module):
    """Estimate sequence-constant segment masses and center ratios."""

    def __init__(
        self,
        local_channels=64,
        global_channels=128,
        head_channels=64,
        global_dilations=(1, 2),
        dropout=0.1,
        causal=False,
        mass_residual_limit=0.2,
        kappa_residual_limit=0.1,
        eps=1e-6,
    ):
        super(GlobalLocalParameterEstimator, self).__init__()
        self.eps = float(eps)
        self.mass_residual_limit = float(mass_residual_limit)
        self.kappa_residual_limit = float(kappa_residual_limit)
        self.local_encoder = LocalPoseEncoder(
            input_channels=34,
            hidden_channels=int(local_channels),
            output_channels=int(local_channels),
        )
        self.global_encoder = GlobalPoseEncoder(
            input_channels=int(local_channels),
            output_channels=int(global_channels),
            dilations=global_dilations,
            dropout=float(dropout),
            causal=bool(causal),
            eps=self.eps,
        )

        fused_channels = int(local_channels) + int(global_channels)
        self.mass_fc1 = nn.Linear(fused_channels, int(head_channels))
        self.mass_fc2 = nn.Linear(int(head_channels), 6)
        self.kappa_fc1 = nn.Linear(fused_channels, int(head_channels))
        self.kappa_fc2 = nn.Linear(int(head_channels), len(KAPPA_GROUP_PRIOR))
        _init_linear(self.mass_fc1)
        _init_linear(self.mass_fc2, final=True)
        _init_linear(self.kappa_fc1)
        _init_linear(self.kappa_fc2, final=True)

        mass_prior = torch.tensor(COCO17_MASS_PRIOR, dtype=torch.float32)
        self.register_buffer('mass_prior', mass_prior / mass_prior.sum())
        self.register_buffer(
            'mass_group_index', torch.tensor(MASS_GROUP_INDEX, dtype=torch.long)
        )
        self.register_buffer(
            'kappa_prior', torch.tensor(KAPPA_PRIOR, dtype=torch.float32)
        )
        self.register_buffer(
            'kappa_group_index', torch.tensor(KAPPA_GROUP_INDEX, dtype=torch.long)
        )

    def forward(self, relative_xy, valid_frames):
        if relative_xy.dim() != 4 or relative_xy.shape[-2:] != (17, 2):
            raise ValueError('relative_xy must have shape [P,T,17,2].')
        persons, time_steps = relative_xy.shape[:2]
        relative_flat = relative_xy.reshape(persons, time_steps, 34)
        local_features = self.local_encoder(relative_flat, valid_frames)
        global_temporal, global_features = self.global_encoder(
            local_features, valid_frames
        )
        local_pooled = _masked_mean_time_last(
            local_features, valid_frames, self.eps
        )
        fused_features = torch.cat((global_features, local_pooled), dim=1)

        mass_group_delta = self.mass_fc2(
            F.relu(self.mass_fc1(fused_features), inplace=True)
        )
        expanded_delta = mass_group_delta.index_select(1, self.mass_group_index)
        mass_logits = torch.log(self.mass_prior.clamp(min=self.eps)).unsqueeze(0)
        mass_logits = mass_logits + self.mass_residual_limit * torch.tanh(expanded_delta)
        mass_weights = F.softmax(mass_logits, dim=1)

        kappa_group_delta = self.kappa_fc2(
            F.relu(self.kappa_fc1(fused_features), inplace=True)
        )
        kappa_delta = kappa_group_delta.index_select(1, self.kappa_group_index)
        kappa = self.kappa_prior.unsqueeze(0)
        kappa = kappa + self.kappa_residual_limit * torch.tanh(kappa_delta)
        kappa = kappa.clamp(min=0.0, max=1.0)

        return {
            'local_features': local_features,  # P,T,64
            'local_pooled': local_pooled,  # P,64
            'global_temporal': global_temporal,  # P,128,T
            'global_features': global_features,  # P,128
            'fused_features': fused_features,  # P,192
            'mass_weights': mass_weights,  # P,10
            'kappa': kappa,  # P,10
        }


class DynamicLambdaEstimator(nn.Module):
    """Estimate one bounded XCoM time coefficient for every frame."""

    def __init__(
        self,
        local_channels=64,
        global_channels=128,
        state_channels=4,
        hidden_channels=64,
        lambda_min=0.1,
        lambda_max=2.0,
        lambda_init=1.0,
    ):
        super(DynamicLambdaEstimator, self).__init__()
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_init = float(lambda_init)
        if not self.lambda_min < self.lambda_max:
            raise ValueError('lambda_min must be smaller than lambda_max.')
        if not self.lambda_min < self.lambda_init < self.lambda_max:
            raise ValueError('lambda_init must lie strictly inside the configured range.')

        input_channels = int(local_channels) + int(global_channels) + int(state_channels)
        self.fc1 = nn.Linear(input_channels, int(hidden_channels))
        self.fc2 = nn.Linear(int(hidden_channels), 1)
        _init_linear(self.fc1)
        _init_linear(self.fc2, final=True)
        initial_ratio = (
            (self.lambda_init - self.lambda_min)
            / (self.lambda_max - self.lambda_min)
        )
        nn.init.constant_(self.fc2.bias, _inverse_sigmoid(initial_ratio))

    def forward(self, global_features, local_features, state_features, valid_frames):
        if global_features.dim() != 2:
            raise ValueError('global_features must have shape [P,Cg].')
        if local_features.dim() != 3 or state_features.dim() != 3:
            raise ValueError('local_features and state_features must have shape [P,T,C].')
        time_steps = local_features.size(1)
        repeated_global = global_features.unsqueeze(1).expand(-1, time_steps, -1)
        inputs = torch.cat((repeated_global, local_features, state_features), dim=-1)
        raw = self.fc2(F.relu(self.fc1(inputs), inplace=True))
        value = self.lambda_min + (
            self.lambda_max - self.lambda_min
        ) * torch.sigmoid(raw)
        fallback = value.new_full(value.shape, self.lambda_init)
        return torch.where(valid_frames.unsqueeze(-1), value, fallback)


class ApparentXCoMExtractor(nn.Module):
    """Build a six-channel XCoM descriptor from absolute and relative COCO17 poses.

    Input uses CTR-GCN layout [N,C,T,17,M]. Only channels 0:4 are read:
    [abs_x, abs_y, rel_x, rel_y]. A fifth score channel may be present for
    CTR-GCN, but it is intentionally ignored by this branch.
    """

    def __init__(
        self,
        min_scale_links=4,
        scale_smooth_window=7,
        scale_ratio_min=0.5,
        scale_ratio_max=2.0,
        mass_residual_limit=0.2,
        kappa_residual_limit=0.1,
        lambda_min=0.1,
        lambda_max=2.0,
        lambda_init=1.0,
        local_channels=64,
        global_channels=128,
        parameter_head_channels=64,
        global_dilations=(1, 2),
        dropout=0.1,
        causal=False,
        velocity_mode='central',
        eps=1e-6,
    ):
        super(ApparentXCoMExtractor, self).__init__()
        if velocity_mode not in ('central', 'backward'):
            raise ValueError("velocity_mode must be 'central' or 'backward'.")
        self.min_scale_links = int(min_scale_links)
        self.scale_smooth_window = max(1, int(scale_smooth_window))
        self.scale_ratio_min = float(scale_ratio_min)
        self.scale_ratio_max = float(scale_ratio_max)
        self.velocity_mode = velocity_mode
        self.eps = float(eps)
        self.lambda_init = float(lambda_init)

        self.register_buffer(
            'scale_links', torch.tensor(COCO17_SCALE_LINKS, dtype=torch.long)
        )
        self.parameter_estimator = GlobalLocalParameterEstimator(
            local_channels=int(local_channels),
            global_channels=int(global_channels),
            head_channels=int(parameter_head_channels),
            global_dilations=global_dilations,
            dropout=float(dropout),
            causal=bool(causal),
            mass_residual_limit=float(mass_residual_limit),
            kappa_residual_limit=float(kappa_residual_limit),
            eps=self.eps,
        )
        self.lambda_estimator = DynamicLambdaEstimator(
            local_channels=int(local_channels),
            global_channels=int(global_channels),
            state_channels=4,
            hidden_channels=int(parameter_head_channels),
            lambda_min=float(lambda_min),
            lambda_max=float(lambda_max),
            lambda_init=self.lambda_init,
        )

    def _sanitize_coordinates(self, absolute_xy, relative_xy):
        finite = (
            torch.isfinite(absolute_xy).all(dim=-1)
            & torch.isfinite(relative_xy).all(dim=-1)
        )
        observed = (
            (absolute_xy.abs().sum(dim=-1) > self.eps)
            | (relative_xy.abs().sum(dim=-1) > self.eps)
        )
        valid = finite & observed
        absolute_xy = torch.where(
            valid.unsqueeze(-1), absolute_xy, torch.zeros_like(absolute_xy)
        )
        relative_xy = torch.where(
            valid.unsqueeze(-1), relative_xy, torch.zeros_like(relative_xy)
        )
        return absolute_xy, relative_xy, valid

    def _group_point(self, xy, valid, indices):
        index = torch.tensor(indices, dtype=torch.long, device=xy.device)
        points = xy.index_select(2, index)
        weights = valid.index_select(2, index).to(xy.dtype)
        denominator = weights.sum(dim=2)
        point = (points * weights.unsqueeze(-1)).sum(dim=2)
        point = point / denominator.clamp(min=self.eps).unsqueeze(-1)
        point_valid = denominator > 0
        point = torch.where(point_valid.unsqueeze(-1), point, torch.zeros_like(point))
        return point, point_valid

    @staticmethod
    def _line_center(proximal, distal, proximal_valid, distal_valid, kappa):
        coefficient = kappa.view(kappa.size(0), 1, 1)
        center = proximal + coefficient * (distal - proximal)
        center_valid = proximal_valid & distal_valid
        center = torch.where(
            center_valid.unsqueeze(-1), center, torch.zeros_like(center)
        )
        return center, center_valid

    def _segment_centers(self, xy, valid, kappa):
        head_landmarks, head_landmarks_valid = self._group_point(
            xy, valid, (0, 1, 2, 3, 4)
        )
        shoulders, shoulders_valid = self._group_point(xy, valid, (5, 6))
        hips, hips_valid = self._group_point(xy, valid, (11, 12))

        head, head_valid = self._line_center(
            shoulders,
            head_landmarks,
            shoulders_valid,
            head_landmarks_valid,
            kappa[:, 0],
        )
        trunk, trunk_valid = self._line_center(
            hips, shoulders, hips_valid, shoulders_valid, kappa[:, 1]
        )
        left_upper, left_upper_valid = self._line_center(
            xy[:, :, 5], xy[:, :, 7], valid[:, :, 5], valid[:, :, 7], kappa[:, 2]
        )
        right_upper, right_upper_valid = self._line_center(
            xy[:, :, 6], xy[:, :, 8], valid[:, :, 6], valid[:, :, 8], kappa[:, 3]
        )
        left_forearm, left_forearm_valid = self._line_center(
            xy[:, :, 7], xy[:, :, 9], valid[:, :, 7], valid[:, :, 9], kappa[:, 4]
        )
        right_forearm, right_forearm_valid = self._line_center(
            xy[:, :, 8], xy[:, :, 10], valid[:, :, 8], valid[:, :, 10], kappa[:, 5]
        )

        forearm_mass = 0.0150
        hand_mass = 0.00585
        left_forearm_hand = (
            forearm_mass * left_forearm + hand_mass * xy[:, :, 9]
        ) / (forearm_mass + hand_mass)
        right_forearm_hand = (
            forearm_mass * right_forearm + hand_mass * xy[:, :, 10]
        ) / (forearm_mass + hand_mass)
        left_forearm_hand_valid = left_forearm_valid & valid[:, :, 9]
        right_forearm_hand_valid = right_forearm_valid & valid[:, :, 10]

        left_thigh, left_thigh_valid = self._line_center(
            xy[:, :, 11], xy[:, :, 13], valid[:, :, 11], valid[:, :, 13], kappa[:, 6]
        )
        right_thigh, right_thigh_valid = self._line_center(
            xy[:, :, 12], xy[:, :, 14], valid[:, :, 12], valid[:, :, 14], kappa[:, 7]
        )
        left_shank, left_shank_valid = self._line_center(
            xy[:, :, 13], xy[:, :, 15], valid[:, :, 13], valid[:, :, 15], kappa[:, 8]
        )
        right_shank, right_shank_valid = self._line_center(
            xy[:, :, 14], xy[:, :, 16], valid[:, :, 14], valid[:, :, 16], kappa[:, 9]
        )

        shank_mass = 0.0457
        foot_mass = 0.0133
        left_shank_foot = (
            shank_mass * left_shank + foot_mass * xy[:, :, 15]
        ) / (shank_mass + foot_mass)
        right_shank_foot = (
            shank_mass * right_shank + foot_mass * xy[:, :, 16]
        ) / (shank_mass + foot_mass)
        left_shank_foot_valid = left_shank_valid & valid[:, :, 15]
        right_shank_foot_valid = right_shank_valid & valid[:, :, 16]

        centers = torch.stack((
            head,
            trunk,
            left_upper,
            right_upper,
            left_forearm_hand,
            right_forearm_hand,
            left_thigh,
            right_thigh,
            left_shank_foot,
            right_shank_foot,
        ), dim=2)
        segment_valid = torch.stack((
            head_valid,
            trunk_valid,
            left_upper_valid,
            right_upper_valid,
            left_forearm_hand_valid,
            right_forearm_hand_valid,
            left_thigh_valid,
            right_thigh_valid,
            left_shank_foot_valid,
            right_shank_foot_valid,
        ), dim=2)
        centers = torch.where(
            segment_valid.unsqueeze(-1), centers, torch.zeros_like(centers)
        )
        return centers, segment_valid

    def _weighted_com(self, centers, segment_valid, mass_weights):
        effective_mass = mass_weights.unsqueeze(1) * segment_valid.to(centers.dtype)
        denominator = effective_mass.sum(dim=2)
        com = (centers * effective_mass.unsqueeze(-1)).sum(dim=2)
        com = com / denominator.clamp(min=self.eps).unsqueeze(-1)
        frame_valid = denominator > self.eps
        com = torch.where(frame_valid.unsqueeze(-1), com, torch.zeros_like(com))
        return com, frame_valid

    def _estimate_scales(self, xy, valid):
        # Scale is deterministic and is not optimized by the classification loss.
        with torch.no_grad():
            first = self.scale_links[:, 0]
            second = self.scale_links[:, 1]
            link_lengths = torch.norm(
                xy.index_select(2, second) - xy.index_select(2, first),
                p=2,
                dim=-1,
            )
            link_valid = valid.index_select(2, first) & valid.index_select(2, second)
            usable_lengths = link_valid & (link_lengths > self.eps)
            nan = link_lengths.new_tensor(float('nan'))
            masked_lengths = torch.where(usable_lengths, link_lengths, nan)
            reference = torch.nanmedian(masked_lengths, dim=1).values
            reference_valid = torch.isfinite(reference) & (reference > self.eps)

            masked_reference = torch.where(reference_valid, reference, nan)
            canonical_scale = torch.nanmedian(masked_reference, dim=1).values

            y = xy[:, :, :, 1]
            positive_inf = y.new_tensor(float('inf'))
            negative_inf = y.new_tensor(float('-inf'))
            frame_min = torch.where(valid, y, positive_inf).amin(dim=2)
            frame_max = torch.where(valid, y, negative_inf).amax(dim=2)
            frame_height = frame_max - frame_min
            valid_height = torch.isfinite(frame_height) & (frame_height > self.eps)
            fallback_height = torch.nanmedian(
                torch.where(valid_height, frame_height, nan), dim=1
            ).values
            fallback_height = torch.where(
                torch.isfinite(fallback_height),
                fallback_height,
                torch.ones_like(fallback_height),
            )
            canonical_scale = torch.where(
                torch.isfinite(canonical_scale), canonical_scale, fallback_height
            ).clamp(min=self.eps)

            ratios = link_lengths / reference.clamp(min=self.eps).unsqueeze(1)
            ratio_valid = usable_lengths & reference_valid.unsqueeze(1)
            ratio_count = ratio_valid.long().sum(dim=2)
            raw_ratio = torch.nanmedian(
                torch.where(ratio_valid, ratios, nan), dim=2
            ).values
            raw_ratio = torch.where(
                ratio_count >= self.min_scale_links, raw_ratio, nan
            )

            global_ratio = torch.nanmedian(raw_ratio, dim=1).values
            global_ratio = torch.where(
                torch.isfinite(global_ratio),
                global_ratio,
                torch.ones_like(global_ratio),
            )

            window = self.scale_smooth_window
            left = window // 2
            right = window - 1 - left
            padded = F.pad(raw_ratio.unsqueeze(1), (left, right), value=float('nan'))
            neighborhoods = padded.unfold(dimension=2, size=window, step=1)
            smooth_ratio = torch.nanmedian(neighborhoods, dim=-1).values.squeeze(1)
            smooth_ratio = torch.where(
                torch.isfinite(smooth_ratio),
                smooth_ratio,
                global_ratio.unsqueeze(1),
            )
            smooth_ratio = smooth_ratio.clamp(
                min=self.scale_ratio_min, max=self.scale_ratio_max
            )
            return (
                canonical_scale.unsqueeze(1) * smooth_ratio
            ).clamp(min=self.eps)

    def _velocity(self, com_abs, frame_valid, frame_interval):
        if frame_interval <= 0:
            raise ValueError('frame_interval must be positive.')
        velocity = torch.zeros_like(com_abs)
        velocity_valid = torch.zeros_like(frame_valid)
        time_steps = com_abs.size(1)
        if time_steps == 1:
            return velocity, velocity_valid

        if self.velocity_mode == 'backward':
            pair_valid = frame_valid[:, 1:] & frame_valid[:, :-1]
            difference = (com_abs[:, 1:] - com_abs[:, :-1]) / frame_interval
            velocity[:, 1:] = torch.where(
                pair_valid.unsqueeze(-1), difference, torch.zeros_like(difference)
            )
            velocity_valid[:, 1:] = pair_valid
            velocity[:, 0] = velocity[:, 1]
            velocity_valid[:, 0] = velocity_valid[:, 1]
            return velocity, velocity_valid

        if time_steps > 2:
            middle_valid = (
                frame_valid[:, 1:-1]
                & frame_valid[:, 2:]
                & frame_valid[:, :-2]
            )
            middle = (
                com_abs[:, 2:] - com_abs[:, :-2]
            ) / (2.0 * frame_interval)
            velocity[:, 1:-1] = torch.where(
                middle_valid.unsqueeze(-1), middle, torch.zeros_like(middle)
            )
            velocity_valid[:, 1:-1] = middle_valid

        first_valid = frame_valid[:, 0] & frame_valid[:, 1]
        first = (com_abs[:, 1] - com_abs[:, 0]) / frame_interval
        velocity[:, 0] = torch.where(
            first_valid.unsqueeze(-1), first, torch.zeros_like(first)
        )
        velocity_valid[:, 0] = first_valid

        last_valid = frame_valid[:, -1] & frame_valid[:, -2]
        last = (com_abs[:, -1] - com_abs[:, -2]) / frame_interval
        velocity[:, -1] = torch.where(
            last_valid.unsqueeze(-1), last, torch.zeros_like(last)
        )
        velocity_valid[:, -1] = last_valid
        return velocity, velocity_valid

    def _dynamic_state(self, com_rel, relative_xy, valid):
        com_height = (-com_rel[:, :, 1:2]).clamp(min=0.0)
        ankle_valid = valid[:, :, 15] & valid[:, :, 16]
        support_width = (
            relative_xy[:, :, 15, 0] - relative_xy[:, :, 16, 0]
        ).abs().unsqueeze(-1)
        support_width = torch.where(
            ankle_valid.unsqueeze(-1), support_width, torch.zeros_like(support_width)
        )
        return torch.cat((com_rel, com_height, support_width), dim=-1)

    def forward(self, skeleton, frame_interval=1.0):
        if skeleton.dim() != 5:
            raise ValueError('skeleton must have shape [N,C,T,V,M].')
        n, channels, time_steps, joints, persons = skeleton.shape
        if channels < 4 or joints != 17:
            raise ValueError('XCoM input requires at least four channels and 17 joints.')

        flat = skeleton[:, :4].permute(0, 4, 2, 3, 1).contiguous()
        flat = flat.view(n * persons, time_steps, joints, 4)
        absolute_xy, relative_xy, valid = self._sanitize_coordinates(
            flat[:, :, :, 0:2], flat[:, :, :, 2:4]
        )
        valid_frames = valid.any(dim=2)

        parameters = self.parameter_estimator(relative_xy, valid_frames)
        mass_weights = parameters['mass_weights']
        kappa = parameters['kappa']

        absolute_centers, absolute_segments_valid = self._segment_centers(
            absolute_xy, valid, kappa
        )
        relative_centers, relative_segments_valid = self._segment_centers(
            relative_xy, valid, kappa
        )
        com_abs, com_abs_valid = self._weighted_com(
            absolute_centers, absolute_segments_valid, mass_weights
        )
        com_rel, com_rel_valid = self._weighted_com(
            relative_centers, relative_segments_valid, mass_weights
        )

        scales = self._estimate_scales(absolute_xy.detach(), valid.detach())
        velocity_abs, velocity_valid = self._velocity(
            com_abs, com_abs_valid, float(frame_interval)
        )
        velocity_norm = velocity_abs / scales.unsqueeze(-1)
        velocity_norm = torch.where(
            velocity_valid.unsqueeze(-1), velocity_norm, torch.zeros_like(velocity_norm)
        )

        state_features = self._dynamic_state(com_rel, relative_xy, valid)
        lambda_value = self.lambda_estimator(
            parameters['global_features'],
            parameters['local_features'],
            state_features,
            com_rel_valid,
        )
        xcom_rel = com_rel + lambda_value * velocity_norm
        xcom_rel = torch.where(
            com_rel_valid.unsqueeze(-1), xcom_rel, torch.zeros_like(xcom_rel)
        )
        z_xcom = torch.cat((com_rel, velocity_norm, xcom_rel), dim=-1)

        def restore(value):
            return value.view(n, persons, *value.shape[1:])

        result = {
            'z_xcom': restore(z_xcom),  # N,M,T,6
            'com_abs': restore(com_abs),  # N,M,T,2
            'com_rel': restore(com_rel),  # N,M,T,2
            'velocity_norm': restore(velocity_norm),  # N,M,T,2
            'xcom_rel': restore(xcom_rel),  # N,M,T,2
            'valid_joints': restore(valid),  # N,M,T,17
            'valid_frames': restore(com_rel_valid),  # N,M,T
            'velocity_valid': restore(velocity_valid),  # N,M,T
            'scales': restore(scales),  # N,M,T
            'mass_weights': restore(mass_weights),  # N,M,10
            'kappa': restore(kappa),  # N,M,10
            'lambda': restore(lambda_value),  # N,M,T,1
            'local_features': restore(parameters['local_features']),  # N,M,T,64
            'local_pooled': restore(parameters['local_pooled']),  # N,M,64
            'global_temporal': restore(parameters['global_temporal']),  # N,M,128,T
            'global_features': restore(parameters['global_features']),  # N,M,128
        }
        return result


class XCoMTemporalBranch(nn.Module):
    """Encode T x 6 XCoM descriptors and align them to CTR-GCN time steps."""

    def __init__(
        self,
        hidden_channels=64,
        output_channels=256,
        dilations=(1, 2, 4),
        dropout=0.1,
        causal=False,
        extractor_args=None,
    ):
        super(XCoMTemporalBranch, self).__init__()
        extractor_args = dict(extractor_args or {})
        extractor_args.setdefault('causal', bool(causal))
        self.extractor = ApparentXCoMExtractor(**extractor_args)

        blocks = []
        in_channels = 6
        for dilation in dilations:
            blocks.append(TemporalResidualBlock(
                in_channels,
                int(hidden_channels),
                dilation=int(dilation),
                dropout=float(dropout),
                causal=bool(causal),
            ))
            in_channels = int(hidden_channels)
        self.temporal_encoder = nn.Sequential(*blocks)
        self.projection = nn.Conv1d(
            int(hidden_channels), int(output_channels), kernel_size=1
        )
        _init_conv(self.projection)

    def forward(self, skeleton, target_length=None, frame_interval=1.0):
        result = self.extractor(skeleton, frame_interval=frame_interval)
        z_xcom = result['z_xcom']
        n, persons, time_steps, channels = z_xcom.shape
        temporal_input = z_xcom.view(n * persons, time_steps, channels)
        temporal_input = temporal_input.permute(0, 2, 1).contiguous()
        encoded = self.temporal_encoder(temporal_input)

        if target_length is None:
            target_length = encoded.size(-1)
        target_length = int(target_length)
        if target_length <= 0:
            raise ValueError('target_length must be positive.')
        aligned = F.adaptive_avg_pool1d(encoded, target_length)
        projected = self.projection(aligned)

        result.update({
            'temporal_features_flat': encoded,  # P,64,T
            'projected_features_flat': projected,  # P,256,T_target
            'temporal_features': encoded.view(
                n, persons, encoded.size(1), encoded.size(2)
            ),
            'projected_features': projected.view(
                n, persons, projected.size(1), projected.size(2)
            ),
        })
        return result


class GatedXCoMFusion(nn.Module):
    """Fuse aligned XCoM and CTR-GCN features before temporal pooling."""

    def __init__(self, channels=256, residual_scale_init=0.0):
        super(GatedXCoMFusion, self).__init__()
        channels = int(channels)
        self.gate = nn.Conv1d(channels * 2, channels, kernel_size=1)
        _init_conv(self.gate)
        nn.init.constant_(self.gate.bias, -2.0)
        self.residual_scale = nn.Parameter(torch.tensor(
            float(residual_scale_init), dtype=torch.float32
        ))

    def forward(self, ctr_features, xcom_features):
        if ctr_features.dim() == 4:
            ctr_temporal = ctr_features.mean(dim=-1)
        elif ctr_features.dim() == 3:
            ctr_temporal = ctr_features
        else:
            raise ValueError('ctr_features must have shape [P,C,T,V] or [P,C,T].')
        if xcom_features.dim() != 3:
            raise ValueError('xcom_features must have shape [P,C,T].')
        if ctr_temporal.size(0) != xcom_features.size(0):
            raise ValueError('CTR-GCN and XCoM person dimensions do not match.')
        if ctr_temporal.size(1) != xcom_features.size(1):
            raise ValueError('CTR-GCN and XCoM channel dimensions do not match.')

        target_length = ctr_temporal.size(-1)
        if xcom_features.size(-1) != target_length:
            xcom_features = F.adaptive_avg_pool1d(xcom_features, target_length)
        gate_input = torch.cat((ctr_temporal, xcom_features), dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused_temporal = (
            ctr_temporal
            + self.residual_scale * gate * xcom_features
        )
        return {
            'fused_temporal': fused_temporal,  # P,C,T
            'ctr_temporal': ctr_temporal,
            'xcom_temporal': xcom_features,
            'gate': gate,
        }

    @staticmethod
    def pool_for_classifier(fused_temporal, batch_size, num_person):
        if fused_temporal.dim() != 3:
            raise ValueError('fused_temporal must have shape [P,C,T].')
        expected_persons = int(batch_size) * int(num_person)
        if fused_temporal.size(0) != expected_persons:
            raise ValueError('P must equal batch_size * num_person.')
        pooled = fused_temporal.mean(dim=-1)
        pooled = pooled.view(int(batch_size), int(num_person), pooled.size(1))
        return pooled.mean(dim=1)
