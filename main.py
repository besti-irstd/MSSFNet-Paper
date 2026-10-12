from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as Data
from model.MSSFNet import MSSFNet
from utils.checkpoint import load_checkpoint, extract_state_dict
from model.loss import *
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time
from utils.tools import *

os.environ['CUDA_VISIBLE_DEVICES']="0"

PROJECT_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = PROJECT_DIR.parent / "MSSFNet-Code"

def parse_args():
    parser = ArgumentParser(description='Train or evaluate MSSFNet')

    parser.add_argument('--dataset-dir', type=str, default=str(ORIGINAL_DIR / 'datasets' / 'IRSTD-1k'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)
    
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Training checkpoint to resume; required with --if-checkpoint')
    # 辅助输出开关；AWFE 在所有配置中均执行。旧参数名作为兼容别名保留。
    parser.add_argument('--deep-supervision', '--use-dhpf', dest='deep_supervision',
                        type=int, default=0, help='Enable auxiliary masks after the warm epoch (0 or 1; default: 0); AWFE always runs; legacy alias: --use-dhpf')

    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--weight-path', type=str, default=str(ORIGINAL_DIR / 'weight' / 'IRSTD-1K' / 'weight.pkl'))

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'
        if args.mode == 'train' and args.if_checkpoint and not args.checkpoint_path:
            raise ValueError('--if-checkpoint requires --checkpoint-path pointing to a training checkpoint.')
        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False)

        device = torch.device('cuda')
        self.device = device

        model = MSSFNet(3)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use ' + str(torch.cuda.device_count()) + ' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 25, eta_min=0.045, last_epoch=-1)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        # Load into the underlying MSSFNet so legacy module names are converted
        # even when the model is wrapped by DataParallel.
        model_for_loading = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if args.mode == 'train':
            if args.if_checkpoint:
                checkpoint = load_checkpoint(args.checkpoint_path, map_location='cpu')
                model_for_loading.load_state_dict(extract_state_dict(checkpoint))
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_iou = checkpoint['iou']
            run_name = 'MSSFNet-%s' % time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            save_folder = PROJECT_DIR / 'weight' / run_name
            suffix = 1
            while save_folder.exists():
                save_folder = PROJECT_DIR / 'weight' / f'{run_name}-{suffix}'
                suffix += 1
            save_folder.mkdir(parents=True)
            self.save_folder = str(save_folder)
        if args.mode == 'test':
            weight = load_checkpoint(args.weight_path, map_location='cpu')
            model_for_loading.load_state_dict(extract_state_dict(weight))
            self.warm_epoch = -1
        

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        
        # tag 在循环内根据 deep_supervision 和 epoch 动态判定
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device)
            labels = mask.to(self.device)

            # 深监督仅在开关启用且 epoch > warm_epoch 时启用；AWFE 始终执行。
            if self.args.deep_supervision and epoch > self.warm_epoch:
                tag = True
            else:
                tag = False

            masks, pred = self.model(data, tag)
            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            for j in range(len(masks)):
                if j>0:
                    labels = self.down(labels)
                loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)
                
            loss = loss / (len(masks)+1)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
        self.scheduler.step()
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        losses = AverageMeter()
        tbar = tqdm(self.val_loader)
        
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
                
                mask = mask.to(self.device)
                data = data.to(self.device)

                # 测试时保留与训练相同的深监督开关判定。
                if self.args.deep_supervision and epoch > self.warm_epoch:
                    tag = True
                else:
                    tag = False

                loss = 0
                _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)
                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()
                # losses.update(loss.item(), pred.size(0))
                tbar.set_description('Epoch %d, IoU %.4f, loss %.4f' % (epoch, mean_IoU, losses.avg))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            if self.mode == 'train':
                with open(osp.join(self.save_folder, 'log.txt'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, mean_IoU, PD[0], FA[0] * 1000000))
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, PD[0], FA[0] * 1000000))
                        
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                torch.save(all_states, self.save_folder+'/checkpoint.pkl')
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')

if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)