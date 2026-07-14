#!/usr/bin/env python
from __future__ import print_function

import argparse
import copy
import csv
import inspect
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import fall_detection
from torchlight import DictAction

try:
    import resource
except ImportError:
    resource = None

if resource is not None:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            'Class %s cannot be found (%s)' % (
                class_str, traceback.format_exception(*sys.exc_info())
            )
        )


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network'
    )
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results'
    )
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file'
    )

    parser.add_argument(
        '--phase',
        default='train',
        choices=['train', 'test', 'model_size'],
        help='must be train, test or model_size'
    )
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if true, the classification score will be stored'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed for pytorch'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)'
    )
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)'
    )
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not'
    )
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown'
    )

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used'
    )
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader'
    )
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training'
    )
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test'
    )

    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model'
    )
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization'
    )
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization'
    )

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduces the learning rate'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing'
    )
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='training batch size'
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size'
    )
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch'
    )
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer'
    )
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate'
    )
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    parser.add_argument(
        '--positive-class-id',
        type=int,
        default=None,
        help='fall positive class id for alarm calibration'
    )
    parser.add_argument(
        '--positive-source-id',
        type=int,
        default=None,
        help='source label id for the fall class before compact remapping'
    )
    parser.add_argument(
        '--fall-like-seed-ids',
        type=int,
        default=[],
        nargs='+',
        help='seed class ids for fall-like analysis'
    )
    parser.add_argument(
        '--fall-like-source-ids',
        type=int,
        default=[],
        nargs='+',
        help='source label ids mapped to fall-like after compact remapping'
    )
    parser.add_argument(
        '--monitored-source-ids',
        type=int,
        default=[],
        nargs='+',
        help='source label ids highlighted in grouped evaluation reports'
    )
    parser.add_argument(
        '--retained-source-ids',
        type=int,
        default=[],
        nargs='+',
        help='source label ids retained in the compact training subset'
    )
    parser.add_argument(
        '--shadow-test-feeder-args',
        action=DictAction,
        default=dict(),
        help='optional feeder args for deleted-class shadow OOD evaluation'
    )
    parser.add_argument(
        '--class-weight-rule',
        action=DictAction,
        default=dict(),
        help='class weighting rule for training'
    )
    parser.add_argument(
        '--oversample-rule',
        action=DictAction,
        default=dict(),
        help='sampling rule for training'
    )
    parser.add_argument(
        '--hard-negative-rule',
        action=DictAction,
        default=dict(),
        help='hard negative mining rule'
    )
    parser.add_argument(
        '--relative-coordinate-rule',
        action=DictAction,
        default=dict(),
        help='five-channel absolute/relative coordinate preprocessing rule'
    )
    return parser


class Processor(object):
    """Processor for skeleton-based action recognition."""

    def __init__(self, arg):
        self.arg = arg
        self.output_device = arg.device[0] if isinstance(arg.device, list) else arg.device
        self.fall_config = fall_detection.build_config(
            vars(arg),
            num_classes=self.arg.model_args.get('num_class')
        )
        raw_label_names = fall_detection.load_label_names(
            os.path.join(os.getcwd(), 'label.txt')
        )
        self.source_label_names = fall_detection.expand_source_label_names(
            raw_label_names,
            self.fall_config
        )
        self.label_names = fall_detection.build_compact_label_names(
            raw_label_names,
            self.fall_config
        )
        self.current_base_lr = self.arg.base_lr
        self.current_steps = list(self.arg.step)
        self.current_warm_up_epoch = self.arg.warm_up_epoch
        self.global_step = 0
        self.best_acc = -1.0
        self.best_acc_epoch = 0
        self.best_stage2_epoch = 0
        self.best_stage2_report = None
        self.best_stage2_sort_key = None
        self.stage1_best_path = os.path.join(self.arg.work_dir, 'best_stage1.pt')
        self.stage2_best_path = os.path.join(self.arg.work_dir, 'best_stage2.pt')
        self.stage1_report_path = os.path.join(self.arg.work_dir, 'fall_detection_stage1.json')
        self.final_report_path = os.path.join(self.arg.work_dir, 'fall_detection_report.json')
        self.hard_negative_path = os.path.join(self.arg.work_dir, 'hard_negative_report.json')
        self.current_stage = 'stage1'

        self.save_arg()
        self._init_writers()
        self.load_model()

        if self.arg.phase != 'model_size':
            self.load_data()

        self.model = self.model.cuda(self.output_device)
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=self.output_device
            )
        if self.arg.phase != 'model_size':
            self.load_optimizer(self.current_base_lr)

        if self.arg.phase == 'test' and self.fall_config['enabled']:
            self.load_existing_fall_metadata()

    def _init_writers(self):
        self.train_writer = None
        self.val_writer = None
        if self.arg.phase != 'train':
            return

        if not self.arg.train_feeder_args.get('debug', False):
            self.arg.model_saved_name = os.path.join(self.arg.work_dir, 'runs')
            if os.path.isdir(self.arg.model_saved_name):
                print('log_dir: ', self.arg.model_saved_name, 'already exist')
                self.handle_existing_log_dir()
            self.train_writer = SummaryWriter(
                os.path.join(self.arg.model_saved_name, 'train'), 'train'
            )
            self.val_writer = SummaryWriter(
                os.path.join(self.arg.model_saved_name, 'val'), 'val'
            )
        else:
            self.arg.model_saved_name = os.path.join(self.arg.work_dir, 'runs')
            self.train_writer = self.val_writer = SummaryWriter(
                os.path.join(self.arg.model_saved_name, 'test'), 'test'
            )

    def handle_existing_log_dir(self):
        backup_dir = self.build_backup_dir(self.arg.model_saved_name)
        shutil.move(self.arg.model_saved_name, backup_dir)
        print('Moved existing log_dir to: ', backup_dir)

    def build_backup_dir(self, dir_path):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        candidate = '{}_backup_{}'.format(dir_path, timestamp)
        suffix = 1
        while os.path.exists(candidate):
            candidate = '{}_backup_{}_{}'.format(dir_path, timestamp, suffix)
            suffix += 1
        return candidate

    def load_existing_fall_metadata(self):
        candidate_paths = []
        if self.arg.weights:
            candidate_paths.append(
                os.path.join(os.path.dirname(os.path.abspath(self.arg.weights)), 'fall_detection_report.json')
            )
        candidate_paths.append(self.final_report_path)

        for path in candidate_paths:
            metadata = fall_detection.load_json(path)
            if not metadata:
                continue
            retained_ids = metadata.get('retained_source_ids') or metadata.get('compact_mapping', {}).get('compact_to_source')
            if retained_ids and list(retained_ids) != self.fall_config['retained_source_ids']:
                self.print_log(
                    'Warning: checkpoint metadata retained_source_ids do not match current config.',
                    print_time=False
                )
            break

    def load_model(self):
        model_class = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        self.print_log(str(model_class), print_time=False)
        self.model = model_class(**self.arg.model_args)
        self.print_log(str(self.model), print_time=False)

        loss_kwargs = {}
        if self.fall_config['enabled'] and self.fall_config['class_weight_rule']['enabled']:
            class_weights = fall_detection.build_class_weights(
                self.arg.model_args['num_class'], self.fall_config
            )
            loss_kwargs['weight'] = torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu'
            )
        self.loss = nn.CrossEntropyLoss(**loss_kwargs).cuda(self.output_device)

        if self.arg.weights:
            self.global_step = self._parse_global_step(self.arg.weights)
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            weights = self._load_checkpoint(self.arg.weights)
            self._load_weights_into_model(weights)

    def _parse_global_step(self, weights_path):
        try:
            return int(os.path.splitext(os.path.basename(weights_path))[0].split('-')[-1])
        except (ValueError, IndexError):
            return 0

    def _load_checkpoint(self, weights_path):
        if weights_path.endswith('.pkl'):
            with open(weights_path, 'rb') as file_obj:
                return pickle.load(file_obj)
        return torch.load(weights_path, map_location='cpu')

    def _load_weights_into_model(self, weights):
        if isinstance(weights, dict) and 'model' in weights:
            weights = weights['model']
        elif isinstance(weights, dict) and 'state_dict' in weights:
            weights = weights['state_dict']
        elif isinstance(weights, dict) and 'model_state_dict' in weights:
            weights = weights['model_state_dict']

        normalized = OrderedDict()
        for key, value in weights.items():
            clean_key = key.split('module.')[-1]
            if clean_key not in normalized:
                normalized[clean_key] = value

        keys = list(normalized.keys())
        for ignore_key in self.arg.ignore_weights:
            for key in keys:
                if ignore_key in key and normalized.pop(key, None) is not None:
                    self.print_log('Successfully removed weights: {}.'.format(key))

        model_state = self.model.state_dict()
        translated = OrderedDict()
        for key, value in normalized.items():
            if key in model_state:
                translated[key] = value
            elif ('module.' + key) in model_state:
                translated['module.' + key] = value
            else:
                translated[key] = value

        try:
            self.model.load_state_dict(translated, strict=True)
        except RuntimeError:
            state = self.model.state_dict()
            missing = list(set(state.keys()).difference(set(translated.keys())))
            if missing:
                self.print_log('Can not find these weights:')
                for item in sorted(missing):
                    self.print_log('  ' + item, print_time=False)
            state.update(translated)
            self.model.load_state_dict(state, strict=False)

    def load_optimizer(self, base_lr):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        self.print_log(
            'using warm up, epoch: {}'.format(self.current_warm_up_epoch)
        )

    def load_data(self):
        feeder_class = import_class(self.arg.feeder)
        self.data_loader = {}
        self.shadow_test_dataset = None

        if self.arg.phase == 'train':
            self.train_dataset = feeder_class(**self.arg.train_feeder_args)
            self.data_loader['train'] = self.build_train_loader(self.train_dataset)
        else:
            self.train_dataset = None

        self.test_dataset = feeder_class(**self.arg.test_feeder_args)
        self.data_loader['test'] = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed
        )

        if self.arg.shadow_test_feeder_args:
            self.shadow_test_dataset = feeder_class(**self.arg.shadow_test_feeder_args)
            self.data_loader['shadow_ood'] = DataLoader(
                dataset=self.shadow_test_dataset,
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed
            )

    def build_train_loader(self, dataset, hard_negative_info=None):
        sampler = None
        shuffle = True
        use_class_oversampling = self.fall_config['oversample_rule']['enabled']
        use_hard_negative_sampling = (
            self.fall_config['hard_negative_rule']['enabled']
            and hard_negative_info is not None
        )
        if self.fall_config['enabled'] and (use_class_oversampling or use_hard_negative_sampling):
            sample_weights = fall_detection.build_sample_weights(
                dataset.label,
                self.fall_config,
                hard_negative_info=hard_negative_info
            )
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=self.arg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.arg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed
        )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w', encoding='utf-8') as file_obj:
            file_obj.write("# command line: {}\n\n".format(' '.join(sys.argv)))
            yaml.safe_dump(arg_dict, file_obj, sort_keys=False)

    def adjust_learning_rate(self, epoch):
        if epoch < self.current_warm_up_epoch and self.current_warm_up_epoch > 0:
            lr = self.current_base_lr * float(epoch + 1) / float(self.current_warm_up_epoch)
        else:
            lr = self.current_base_lr * (
                self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.current_steps))
            )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def print_log(self, message, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            message = "[ " + localtime + ' ] ' + message
        print(message)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a', encoding='utf-8') as file_obj:
                print(message, file=file_obj)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_checkpoint(self, file_name):
        state_dict = self.model.state_dict()
        weights = OrderedDict(
            [[key.split('module.')[-1], value.cpu()] for key, value in state_dict.items()]
        )
        path = os.path.join(self.arg.work_dir, file_name)
        torch.save(weights, path)
        return path

    def train_epoch(self, epoch, stage_name, save_model=False, checkpoint_prefix='runs'):
        self.model.train()
        self.print_log('{} training epoch: {}'.format(stage_name, epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        if self.train_writer is not None:
            self.train_writer.add_scalar(stage_name + '/epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            del index
            del batch_idx
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            output = self.model(data)
            loss = self.loss(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            _, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            if self.train_writer is not None:
                self.train_writer.add_scalar(stage_name + '/acc', acc, self.global_step)
                self.train_writer.add_scalar(stage_name + '/loss', loss.data.item(), self.global_step)
                self.train_writer.add_scalar(
                    stage_name + '/lr',
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step
                )
            timer['statistics'] += self.split_time()

        proportion = {
            key: '{:02d}%'.format(int(round(value * 100 / sum(timer.values()))))
            for key, value in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}. Mean training acc: {:.2f}%.'.format(
                float(np.mean(loss_value)), float(np.mean(acc_value) * 100.0)
            )
        )
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion)
        )

        if save_model:
            return self.save_checkpoint(
                '{}-{}-{}.pt'.format(checkpoint_prefix, epoch + 1, int(self.global_step))
            )
        return None

    def evaluate_loader(self, loader_name, loader, dataset, epoch, save_score=False, score_tag=''):
        self.model.eval()
        loss_value = []
        score_frag = []
        label_list = []
        pred_list = []
        index_list = []
        is_shadow_ood = bool(getattr(dataset, 'shadow_ood', False))

        process = tqdm(loader, ncols=40)
        for data, label, index in process:
            label_list.append(label.numpy())
            index_list.append(index.numpy())
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output = self.model(data)
                score_frag.append(output.data.cpu().numpy())
                _, predict_label = torch.max(output.data, 1)
                pred_list.append(predict_label.data.cpu().numpy())
                if not is_shadow_ood:
                    loss = self.loss(output, label)
                    loss_value.append(loss.data.item())

        score = np.concatenate(score_frag) if score_frag else np.empty((0, self.arg.model_args['num_class']))
        label_array = np.concatenate(label_list) if label_list else np.empty((0,), dtype=np.int64)
        pred_array = np.concatenate(pred_list) if pred_list else np.empty((0,), dtype=np.int64)
        index_array = np.concatenate(index_list) if index_list else np.empty((0,), dtype=np.int64)
        loss = float(np.mean(loss_value)) if loss_value else 0.0

        if 'ucla' in self.arg.feeder:
            sample_names = list(np.arange(len(score)))
        else:
            sample_names = list(getattr(dataset, 'sample_name', np.arange(len(score))))
        accuracy = dataset.top_k(score, 1) if len(score) and not is_shadow_ood else 0.0

        score_dict = dict(zip(sample_names, score))
        if save_score:
            score_file = '{}epoch{}_{}_score.pkl'.format(score_tag, epoch + 1, loader_name)
            with open(os.path.join(self.arg.work_dir, score_file), 'wb') as file_obj:
                pickle.dump(score_dict, file_obj)

        if not is_shadow_ood:
            confusion = confusion_matrix(
                label_array,
                pred_array,
                labels=list(range(self.arg.model_args['num_class']))
            )
            list_diag = np.diag(confusion).astype(np.float32)
            list_raw_sum = np.sum(confusion, axis=1).astype(np.float32)
            each_acc = np.divide(
                list_diag,
                np.maximum(list_raw_sum, 1.0),
                out=np.zeros_like(list_diag),
                where=list_raw_sum > 0
            )
            with open(
                os.path.join(self.arg.work_dir, '{}epoch{}_{}_each_class_acc.csv'.format(score_tag, epoch + 1, loader_name)),
                'w',
                newline='',
                encoding='utf-8'
            ) as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(each_acc)
                writer.writerows(confusion)

        result = {
            'loss': loss,
            'score': score,
            'labels': label_array,
            'preds': pred_array,
            'indices': index_array,
            'accuracy': float(accuracy),
            'sample_names': sample_names,
        }
        if self.fall_config['enabled'] and len(score):
            if is_shadow_ood:
                source_labels = getattr(dataset, 'source_label', np.empty((0,), dtype=np.int64))
                source_labels = np.asarray(source_labels, dtype=np.int64)[index_array]
                result['fall_report'] = fall_detection.generate_shadow_ood_report(
                    score,
                    source_labels,
                    self.fall_config,
                    label_names=self.source_label_names
                )
                result['source_labels'] = source_labels
            else:
                result['fall_report'] = fall_detection.generate_report(
                    score,
                    label_array,
                    self.fall_config,
                    label_names=self.source_label_names
                )
        return result

    def eval(self, epoch, save_score=False, loader_names=None, score_tag=''):
        if loader_names is None:
            loader_names = ['test']
            if 'shadow_ood' in self.data_loader:
                loader_names.append('shadow_ood')
        self.print_log('{} eval epoch: {}'.format(self.current_stage, epoch + 1))
        results = {}
        for loader_name in loader_names:
            result = self.evaluate_loader(
                loader_name,
                self.data_loader[loader_name],
                self.data_loader[loader_name].dataset,
                epoch=epoch,
                save_score=save_score,
                score_tag=score_tag
            )
            results[loader_name] = result

            self.print_log(
                '\tMean {} loss of {} batches: {}.'.format(
                    loader_name, len(self.data_loader[loader_name]), result['loss']
                )
            )
            if not getattr(self.data_loader[loader_name].dataset, 'shadow_ood', False):
                for topk in self.arg.show_topk:
                    topk_acc = self.data_loader[loader_name].dataset.top_k(result['score'], topk)
                    self.print_log('\tTop{}: {:.2f}%'.format(topk, 100.0 * topk_acc))

            if self.arg.phase == 'train' and loader_name == 'test' and self.val_writer is not None:
                self.val_writer.add_scalar(self.current_stage + '/loss', result['loss'], self.global_step)
                self.val_writer.add_scalar(self.current_stage + '/acc', result['accuracy'], self.global_step)

            if 'fall_report' in result:
                metrics = result['fall_report']['metrics']
                if loader_name == 'shadow_ood':
                    self.print_log(
                        '\tShadow OOD: fall_rate={:.4f}, fall_like_rate={:.4f}, normal_rate={:.4f}'.format(
                            metrics['predicted_fall_rate'],
                            metrics['predicted_fall_like_rate'],
                            metrics['predicted_normal_rate'],
                        )
                    )
                else:
                    self.print_log(
                        '\tGrouped metrics: fall_precision={:.4f}, fall_recall={:.4f}, '
                        'normal->fall={:.4f}, normal->fall-like={:.4f}, fall-like->fall={:.4f}'.format(
                            metrics['fall_precision'],
                            metrics['fall_recall'],
                            metrics['normal_to_fall_rate'],
                            metrics['normal_to_fall_like_rate'],
                            metrics['fall_like_to_fall_rate'],
                        )
                    )

        return results

    def build_combined_report(self, test_result, shadow_result=None, stage_name=None):
        report = copy.deepcopy(test_result.get('fall_report', {}))
        if not report:
            return report
        if shadow_result is not None and 'fall_report' in shadow_result:
            report['shadow_ood'] = shadow_result['fall_report']
        if stage_name is not None:
            report['stage_name'] = stage_name
        return report

    def build_hard_negative_loader(self):
        feeder_class = import_class(self.arg.feeder)
        feeder_args = copy.deepcopy(self.arg.train_feeder_args)
        feeder_args['split'] = 'train'
        if 'p_interval' in feeder_args:
            feeder_args['p_interval'] = [0.95]
        for key, value in (
            ('random_rot', False),
            ('coord_jitter_sigma', 0.0),
            ('joint_dropout_prob', 0.0),
            ('score_scale_range', [1.0, 1.0]),
        ):
            if key in feeder_args:
                feeder_args[key] = value
        dataset = feeder_class(**feeder_args)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed
        )
        return dataset, loader

    def mine_hard_negatives(self):
        dataset, loader = self.build_hard_negative_loader()
        result = self.evaluate_loader(
            loader_name='train_hard_negative',
            loader=loader,
            dataset=dataset,
            epoch=0,
            save_score=False,
            score_tag='hard_negative_'
        )
        hard_negative_info = fall_detection.find_hard_negative_info(
            result['score'],
            result['labels'],
            self.fall_config
        )
        all_indices = hard_negative_info['all_indices'].tolist()
        fall_indices = hard_negative_info['fall_indices'].tolist()
        fall_like_indices = hard_negative_info['fall_like_indices'].tolist()
        report = {
            'count': int(len(all_indices)),
            'fall_count': int(len(fall_indices)),
            'fall_like_count': int(len(fall_like_indices)),
            'indices': [int(index) for index in all_indices],
            'fall_indices': [int(index) for index in fall_indices],
            'fall_like_indices': [int(index) for index in fall_like_indices],
            'sample_names': [result['sample_names'][index] for index in all_indices],
        }
        fall_detection.save_json(self.hard_negative_path, report)
        return hard_negative_info

    def run_stage1(self):
        self.current_stage = 'stage1'
        self.current_base_lr = self.arg.base_lr
        self.current_steps = list(self.arg.step)
        self.current_warm_up_epoch = self.arg.warm_up_epoch
        self.global_step = int(self.arg.start_epoch * len(self.data_loader['train']))

        self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log('# Parameters: {}'.format(num_params))
        eval_loader_names = ['test']
        if 'shadow_ood' in self.data_loader:
            eval_loader_names.append('shadow_ood')

        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            save_model = (
                ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
            ) and (epoch + 1) > self.arg.save_epoch
            self.train_epoch(epoch, stage_name='stage1', save_model=save_model, checkpoint_prefix='runs')
            eval_results = self.eval(epoch, save_score=False, loader_names=eval_loader_names)
            test_result = eval_results['test']

            if test_result['accuracy'] > self.best_acc:
                self.best_acc = test_result['accuracy']
                self.best_acc_epoch = epoch + 1
                self.save_checkpoint('best_stage1.pt')

        self._load_weights_into_model(torch.load(self.stage1_best_path, map_location='cpu'))
        stage1_eval_results = self.eval(
            epoch=0,
            save_score=True,
            loader_names=eval_loader_names,
            score_tag='stage1_'
        )
        stage1_results = stage1_eval_results['test']
        stage1_report = self.build_combined_report(
            stage1_results,
            shadow_result=stage1_eval_results.get('shadow_ood'),
            stage_name='stage1'
        )
        fall_detection.save_json(self.stage1_report_path, stage1_report)
        return stage1_results, stage1_report

    def run_stage2(self):
        if not (self.fall_config['enabled'] and self.fall_config['hard_negative_rule']['enabled']):
            return None

        hard_negative_info = self.mine_hard_negatives()
        if len(hard_negative_info['all_indices']) == 0:
            self.print_log('No hard negatives found. Skip stage2 fine-tune.')
            return None

        self.current_stage = 'stage2'
        self.data_loader['train'] = self.build_train_loader(
            self.train_dataset,
            hard_negative_info=hard_negative_info
        )
        self.current_base_lr = self.arg.base_lr * float(self.fall_config['hard_negative_rule']['lr_scale'])
        self.current_steps = []
        self.current_warm_up_epoch = 0
        self.load_optimizer(self.current_base_lr)

        fine_tune_epochs = int(self.fall_config['hard_negative_rule']['fine_tune_epochs'])
        self.best_stage2_report = None
        self.best_stage2_sort_key = None
        self.best_stage2_epoch = 0
        eval_loader_names = ['test']
        if 'shadow_ood' in self.data_loader:
            eval_loader_names.append('shadow_ood')

        for epoch in range(fine_tune_epochs):
            self.train_epoch(
                epoch,
                stage_name='stage2',
                save_model=True,
                checkpoint_prefix='stage2-runs'
            )
            eval_results = self.eval(epoch, save_score=False, loader_names=eval_loader_names)
            test_report = eval_results['test'].get('fall_report')
            if test_report is None:
                continue

            shadow_report = eval_results.get('shadow_ood', {}).get('fall_report')
            sort_key = fall_detection.report_sort_key(test_report, shadow_report=shadow_report)
            if self.best_stage2_sort_key is None or sort_key > self.best_stage2_sort_key:
                self.best_stage2_sort_key = sort_key
                self.best_stage2_report = self.build_combined_report(
                    eval_results['test'],
                    shadow_result=eval_results.get('shadow_ood'),
                    stage_name='stage2'
                )
                self.best_stage2_epoch = epoch + 1
                self.save_checkpoint('best_stage2.pt')

        if self.best_stage2_sort_key is None:
            return None

        self._load_weights_into_model(torch.load(self.stage2_best_path, map_location='cpu'))
        final_eval_results = self.eval(
            epoch=0,
            save_score=True,
            loader_names=eval_loader_names,
            score_tag=''
        )
        final_results = final_eval_results['test']
        final_report = self.build_combined_report(
            final_results,
            shadow_result=final_eval_results.get('shadow_ood'),
            stage_name='stage2_final'
        )
        final_report['stage1_report_path'] = self.stage1_report_path
        final_report['hard_negative_report_path'] = self.hard_negative_path
        final_report['stage2_best_epoch'] = self.best_stage2_epoch
        fall_detection.save_json(self.final_report_path, final_report)
        return final_results, final_report

    def summarize_training(self, final_report):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log('Best stage1 accuracy: {}'.format(self.best_acc))
        self.print_log('Best stage1 epoch: {}'.format(self.best_acc_epoch))
        if final_report is not None:
            metrics = final_report['metrics']
            self.print_log(
                'Final grouped metrics: fall_precision={:.4f}, fall_recall={:.4f}, '
                'normal->fall={:.4f}, normal->fall-like={:.4f}, fall-like->fall={:.4f}'.format(
                    metrics['fall_precision'],
                    metrics['fall_recall'],
                    metrics['normal_to_fall_rate'],
                    metrics['normal_to_fall_like_rate'],
                    metrics['fall_like_to_fall_rate']
                )
            )
            shadow_metrics = final_report.get('shadow_ood', {}).get('metrics')
            if shadow_metrics:
                self.print_log(
                    'Shadow OOD: fall_rate={:.4f}, fall_like_rate={:.4f}, normal_rate={:.4f}'.format(
                        shadow_metrics['predicted_fall_rate'],
                        shadow_metrics['predicted_fall_like_rate'],
                        shadow_metrics['predicted_normal_rate']
                    )
                )
        self.print_log('Model name: {}'.format(self.arg.work_dir))
        self.print_log('Model total number of params: {}'.format(num_params))
        self.print_log('Weight decay: {}'.format(self.arg.weight_decay))
        self.print_log('Base LR: {}'.format(self.arg.base_lr))
        self.print_log('Batch Size: {}'.format(self.arg.batch_size))
        self.print_log('Test Batch Size: {}'.format(self.arg.test_batch_size))
        self.print_log('seed: {}'.format(self.arg.seed))

    def start(self):
        if self.arg.phase == 'train':
            _stage1_results, stage1_report = self.run_stage1()
            stage2_output = self.run_stage2()
            final_report = None
            if stage2_output is not None:
                _final_results, final_report = stage2_output
            else:
                final_report = stage1_report
                fall_detection.save_json(self.final_report_path, final_report)
            self.summarize_training(final_report)

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.current_stage = 'test'
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            eval_loader_names = ['test']
            if 'shadow_ood' in self.data_loader:
                eval_loader_names.append('shadow_ood')
            results = self.eval(epoch=0, save_score=self.arg.save_score, loader_names=eval_loader_names)
            if 'fall_report' in results['test']:
                final_report = self.build_combined_report(
                    results['test'],
                    shadow_result=results.get('shadow_ood'),
                    stage_name='test'
                )
                fall_detection.save_json(self.final_report_path, final_report)
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    preview_arg = parser.parse_args()
    if preview_arg.config is not None:
        with open(preview_arg.config, 'r', encoding='utf-8') as file_obj:
            default_arg = yaml.safe_load(file_obj)
        key = vars(preview_arg).keys()
        for config_key in default_arg.keys():
            if config_key not in key:
                print('WRONG ARG: {}'.format(config_key))
                assert config_key in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
