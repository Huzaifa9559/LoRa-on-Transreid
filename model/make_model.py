import copy

import torch
import torch.nn as nn

from config.peft_config import get_lora_blocks, get_peft_method, get_ssf_blocks
from loss.metric_learning import Arcface, AMSoftmax, CircleLoss, Cosface
from model.peft.lightweight import (
    BottleneckAdapter,
    inject_adapters_into_vit,
    mark_trainable_adapters_and_head,
    mark_trainable_bitfit,
    mark_trainable_lntune,
)
from model.peft.lora import (
    inject_lora_into_vit,
    load_lora_state_dict,
    load_prediction_state,
    mark_trainable_lora_and_head,
    maybe_merge_lora,
)

from .backbones.resnet import Bottleneck, ResNet
from .backbones.vit_pytorch import (
    deit_small_patch16_224_TransReID,
    vit_base_patch16_224_TransReID,
    vit_small_patch16_224_TransReID,
)


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except Exception:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)
    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _freeze_non_ssf(named_params):
    for n, p in named_params:
        if 'ssf' not in n:
            p.requires_grad = False


def _load_checkpoint_state(module, trained_path):
    param_dict = torch.load(trained_path, map_location='cpu')
    if isinstance(param_dict, dict) and 'adapters' in param_dict:
        missing, unexpected = load_lora_state_dict(
            module, param_dict['adapters'], strict=False
        )
        if 'prediction_state' in param_dict:
            load_prediction_state(module, param_dict['prediction_state'], strict=False)
        meta = param_dict.get('meta', {})
        print(
            f"Loaded LoRA adapter export from {trained_path} "
            f"(meta={meta}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return
    if isinstance(param_dict, dict) and 'state_dict' in param_dict:
        param_dict = param_dict['state_dict']
    own = module.state_dict()
    for key, value in param_dict.items():
        key = key.replace('module.', '')
        if key not in own:
            continue
        if tuple(own[key].shape) != tuple(value.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: model {tuple(own[key].shape)} "
                f"vs checkpoint {tuple(value.shape)}"
            )
        own[key].copy_(value)
    print('Loading pretrained model from {}'.format(trained_path))


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        if self.neck_feat == 'after':
            return feat
        return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        peft_method = get_peft_method(cfg)

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        camera_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        view_num = view_num if cfg.MODEL.SIE_VIEW else 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            ssf_enabled=(peft_method == 'ssf'),
            ssf_blocks=get_ssf_blocks(cfg),
        )

        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet' and model_path:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        if peft_method == 'ssf' and cfg.PEFT.SSF.FREEZE_BACKBONE:
            _freeze_non_ssf(self.base.named_parameters())
            print('Backbone frozen — only SSF parameters are trainable in self.base')

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        if self.neck == 'no':
            feat = global_feat
        else:
            feat = self.bottleneck(global_feat)
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        if self.neck_feat == 'after':
            return feat
        return global_feat

    def load_param(self, trained_path):
        _load_checkpoint_state(self, trained_path)

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        peft_method = get_peft_method(cfg)

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        camera_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        view_num = view_num if cfg.MODEL.SIE_VIEW else 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            ssf_enabled=(peft_method == 'ssf'),
            ssf_blocks=get_ssf_blocks(cfg),
        )

        if pretrain_choice == 'imagenet' and model_path:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        if peft_method == 'ssf' and cfg.PEFT.SSF.FREEZE_BACKBONE:
            _freeze_non_ssf(self.base.named_parameters())
            print('Backbone frozen — only SSF parameters are trainable in self.base')

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        if peft_method == 'ssf' and cfg.PEFT.SSF.FREEZE_BACKBONE:
            _freeze_non_ssf(self.b1.named_parameters())
            _freeze_non_ssf(self.b2.named_parameters())
            print('JPM branches (b1, b2) frozen — only SSF parameters trainable')

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        self.shift_num = cfg.MODEL.SHIFT_NUM
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label=None, view_label=None):
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        # global branch
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]

        # lf_1
        b1_local_feat = self.b2(torch.cat((token, x[:, :patch_length]), dim=1))
        local_feat_1 = b1_local_feat[:, 0]
        # lf_2
        b2_local_feat = self.b2(torch.cat((token, x[:, patch_length:patch_length * 2]), dim=1))
        local_feat_2 = b2_local_feat[:, 0]
        # lf_3
        b3_local_feat = self.b2(torch.cat((token, x[:, patch_length * 2:patch_length * 3]), dim=1))
        local_feat_3 = b3_local_feat[:, 0]
        # lf_4
        b4_local_feat = self.b2(torch.cat((token, x[:, patch_length * 3:patch_length * 4]), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], [
                global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4
            ]
        if self.neck_feat == 'after':
            return torch.cat(
                [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1
            )
        return torch.cat(
            [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1
        )

    def load_param(self, trained_path):
        _load_checkpoint_state(self, trained_path)

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
}


def _print_trainable_stats(model: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f'Trainable params: {trainable_params:,} / Total params: {total_params:,} '
        f'({100 * trainable_params / total_params:.2f}%)'
    )


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(
                num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE
            )
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')

    peft_method = get_peft_method(cfg)
    if peft_method == 'lora' and cfg.MODEL.NAME == 'transformer':
        print('===========Injecting LoRA adapters===========')
        include_blocks = get_lora_blocks(cfg)
        if include_blocks:
            print(f'Applying LoRA to blocks: {include_blocks}')
        else:
            print('Applying LoRA to all blocks')

        replaced = inject_lora_into_vit(
            model.base,
            r=cfg.PEFT.LORA.R,
            alpha=cfg.PEFT.LORA.ALPHA,
            dropout=cfg.PEFT.LORA.DROPOUT,
            targets=list(cfg.PEFT.LORA.TARGETS),
            bias_mode=cfg.PEFT.LORA.BIAS,
            include_blocks=include_blocks,
        )
        print(
            f'LoRA applied to {len(replaced)} layers: {replaced[:3]}...'
            if len(replaced) > 3 else f'LoRA applied to {len(replaced)} layers: {replaced}'
        )

        mark_trainable_lora_and_head(model, train_head=cfg.PEFT.LORA.TRAIN_HEAD)

        _print_trainable_stats(model)

        maybe_merge_lora(model.base, enabled=True, merge_at_eval=cfg.PEFT.LORA.MERGE_AT_EVAL)
        if cfg.PEFT.LORA.MERGE_AT_EVAL:
            print('LoRA will merge into base weights at eval time')

    elif peft_method == 'bitfit' and cfg.MODEL.NAME == 'transformer':
        print('===========Applying BitFit (bias-only fine-tuning)===========')
        mark_trainable_bitfit(model, blocks=list(cfg.PEFT.BITFIT.BLOCKS) or None, train_head=True)
        _print_trainable_stats(model)

    elif peft_method == 'lntune' and cfg.MODEL.NAME == 'transformer':
        print('===========Applying LN-tuning (LayerNorm-only fine-tuning)===========')
        mark_trainable_lntune(
            model,
            blocks=list(cfg.PEFT.LNTUNE.BLOCKS) or None,
            train_head=True,
            train_final_norm=cfg.PEFT.LNTUNE.TRAIN_FINAL_NORM,
        )
        _print_trainable_stats(model)

    elif peft_method == 'adapter' and cfg.MODEL.NAME == 'transformer':
        print('===========Injecting bottleneck adapters===========')
        include_blocks = list(cfg.PEFT.ADAPTER.BLOCKS) or None
        if include_blocks:
            print(f'Applying adapters to blocks: {include_blocks}')
        else:
            print('Applying adapters to all blocks')

        replaced = inject_adapters_into_vit(
            model.base,
            r=cfg.PEFT.ADAPTER.R,
            dropout=cfg.PEFT.ADAPTER.DROPOUT,
            scale=cfg.PEFT.ADAPTER.SCALE,
            targets=list(cfg.PEFT.ADAPTER.TARGETS),
            include_blocks=include_blocks,
        )
        print(
            f'Adapters applied to {len(replaced)} layers: {replaced[:3]}...'
            if len(replaced) > 3 else f'Adapters applied to {len(replaced)} layers: {replaced}'
        )

        mark_trainable_adapters_and_head(model, train_head=True)
        _print_trainable_stats(model)

    return model
