import os
import utils
import shutil
import logging
import argparse
import importlib
import time
import torch
import torch.distributed as dist
from datetime import datetime
from mmcv.utils import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, build_optimizer, load_checkpoint, init_dist, get_dist_info
from mmdet.apis import set_random_seed
from mmdet.core import DistEvalHook, EvalHook
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from loaders.builder import build_dataloader
from os import path as osp

class MaxIterEpochBasedRunner(EpochBasedRunner):
    """Epoch runner with an optional max_iters early-stop for debug."""
    def __init__(self, *args, max_iters=None, rank_debug_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._user_max_iters = max_iters
        self._rank_debug_interval = rank_debug_interval

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        reached_max_iters = False
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            if self._rank_debug_interval and (self._iter % self._rank_debug_interval == 0):
                try:
                    rank, _ = get_dist_info()
                except Exception:
                    rank = 0
                loss_val = None
                if isinstance(self.outputs, dict) and 'loss' in self.outputs:
                    try:
                        loss_val = float(self.outputs['loss'])
                    except Exception:
                        loss_val = None
                print(f"[Rank {rank}] iter={self._iter} loss={loss_val}")
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1
            cur_iter = self._iter
            if self._user_max_iters is not None and cur_iter >= self._user_max_iters:
                self.logger.info('Max iters reached (%d), stopping early.', self._user_max_iters)
                reached_max_iters = True
                break

        self.call_hook('after_train_epoch')
        self._epoch += 1
        if reached_max_iters:
            self._max_epochs = self._epoch

def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Stop training after N iterations (debug only).')
    parser.add_argument('--rank_debug_interval', type=int, default=0,
                        help='Print per-rank debug log every N iters (0 disables).')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmdet3d'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank, rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    if rank == 0:
        # resume or start a new run
        if cfgs.resume_from is not None:
            assert os.path.isfile(cfgs.resume_from)
            work_dir = os.path.dirname(cfgs.resume_from)
        else:
            run_name = ''
            # if not cfgs.debug:
            #     run_name = input('Name your run (leave blank for default): ')
            if run_name == '':
                run_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            cfgs.work_dir = osp.join(osp.splitext(osp.basename(args.config))[0])
            # work_dir = os.path.join('outputs', cfgs.model.type, run_name)
            work_dir = os.path.join('outputs', cfgs.work_dir, run_name)
            if os.path.exists(work_dir):  # must be an empty dir
                if input('Path "%s" already exists, overwrite it? [Y/n] ' % work_dir) == 'n':
                    print('Bye.')
                    exit(0)
                shutil.rmtree(work_dir)

            os.makedirs(work_dir, exist_ok=False)

        # init logging, backup code
        utils.init_logging(os.path.join(work_dir, 'train.log'), cfgs.debug)
        utils.backup_code(work_dir)
        logging.info('Logs will be saved to %s' % work_dir)

    else:
        # disable logging on other workers
        logging.root.disabled = True
        work_dir = '/tmp'

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist_params = cfgs.get('dist_params', {})
        dist_params.setdefault('backend', 'nccl')
        init_dist('pytorch', **dist_params)
        rank, world_size = get_dist_info()

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)

    logging.info('Loading training set from %s' % cfgs.dataset_root)
    train_dataset = build_dataset(cfgs.data.train)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfgs.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=True,
        seed=0,
    )

    logging.info('Loading validation set from %s' % cfgs.dataset_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.init_weights()

    sync_bn = cfgs.get('sync_bn', False)
    if world_size > 1 and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Convert to SyncBatchNorm')

    logging.info(f'Model:\n{model}')
    
    model.cuda()
    model.train()

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))
    logging.info('Batch size per GPU: %d' % (cfgs.batch_size))

    if world_size > 1:
        find_unused_parameters = cfgs.get('find_unused_parameters', False)
        broadcast_buffers = cfgs.get('broadcast_buffers', False)
        static_graph = cfgs.get('static_graph', False)  # 设置为 True 可解决 checkpoint + DDP 兼容性问题
        model = MMDistributedDataParallel(
            model, [local_rank],
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters)
        # 启用 static_graph 模式，解决 checkpoint + DDP "mark ready twice" 问题
        if static_graph:
            model._set_static_graph()
            logging.info('DDP static_graph enabled')
    else:
        model = MMDataParallel(model, [0])

    logging.info('Creating optimizer: %s' % cfgs.optimizer.type)
    optimizer = build_optimizer(model, cfgs.optimizer)

    if args.max_iters is not None:
        logging.info('Max iters (debug): %d', args.max_iters)

    runner = MaxIterEpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=work_dir,
        logger=logging.root,
        max_epochs=cfgs.total_epochs,
        meta=dict(),
        max_iters=args.max_iters,
        rank_debug_interval=args.rank_debug_interval,
    )

    runner.register_timer_hook(dict(type='IterTimerHook'))
    # register hooks
    runner.register_training_hooks(
        cfgs.lr_config,
        cfgs.optimizer_config,
        cfgs.checkpoint_config,
        cfgs.log_config,
        cfgs.get('momentum_config', None),
        custom_hooks_config=cfgs.get('custom_hooks', None))

    runner.register_custom_hooks(dict(type='DistSamplerSeedHook'))

    if cfgs.eval_config['interval'] > 0:
        if world_size > 1:
            runner.register_hook(DistEvalHook(val_loader, interval=cfgs.eval_config['interval'], gpu_collect=True))
        else:
            runner.register_hook(EvalHook(val_loader, interval=cfgs.eval_config['interval']))

    if cfgs.resume_from is not None:
        logging.info('Resuming from %s' % cfgs.resume_from)
        runner.resume(cfgs.resume_from)

    elif cfgs.load_from is not None:
        logging.info('Loading checkpoint from %s' % cfgs.load_from)
        if cfgs.revise_keys is not None:
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
                revise_keys=cfgs.revise_keys
            )
        else:
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
            )

    runner.run([train_loader], [('train', 1)])


if __name__ == '__main__':
    main()
