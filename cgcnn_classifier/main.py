import argparse
import sys
import os
import shutil
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from random import sample

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from cgcnn.model import CrystalGraphConvNet
from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--root', default='./data/', metavar='DATA_ROOT', 
                    help='path to data root dir')
parser.add_argument('--target', default='MIT', metavar='TARGET_PROPERTY',
                    help="target property ('MIT', 'band_gap', 'energy_per_atom', \
                                           'formation_energy_per_atom')")
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='classification', help='complete a regression or '
                    'classification task (default: classification)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 150)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                    '0.01)')
parser.add_argument('--lr-milestones', default=[30, 60], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                    '[30, 60])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--train-ratio', default=0.7, type=float, metavar='n/N',
                    help='ratio of training data (default: 0.7)')
parser.add_argument('--val-ratio', default=0.15, type=float, metavar='n/N',
                    help='ratio of validation data (default: 0.15)')
parser.add_argument('--test-ratio', default=0.15, type=float, metavar='n/N',
                    help='ratio of test data (default: 0.15)')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

def main():
    global args, best_mae_error

    # load dataset: (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
    dataset = CIFData(args.root+args.target)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset, collate_fn=collate_fn, batch_size=args.batch_size,
        train_ratio=args.train_ratio, num_workers=args.workers,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        pin_memory=args.cuda, return_test=True)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        sample_data_list = [dataset[i] for i in \
                            sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=True if args.task ==
                                'classification' else False)
    # pring number of trainable model parameters
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print('=> number of trainable model parameters: {:d}'.format(trainable_params))

    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # TensorBoard writer
    summary_root = './runs/'
    if not os.path.exists(summary_root):
        os.mkdir(summary_root)
    summary_file = summary_root + args.target
    if os.path.exists(summary_file):
        shutil.rmtree(summary_file)
    writer = SummaryWriter(summary_file)

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer, writer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, epoch, normalizer, writer)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, args.target, is_best)


def train(train_loader, model, criterion, optimizer, epoch, normalizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    running_loss = 0.0
    for i, (features, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        # compute output
        output = model(features[0], features[1], features[2], features[3])
        loss = criterion(output, target_normed)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output), target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error.item(), target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output, target)
            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy.item(), target.size(0))
            precisions.update(precision.item(), target.size(0))
            recalls.update(recall.item(), target.size(0))
            fscores.update(fscore.item(), target.size(0))
            auc_scores.update(auc_score.item(), target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to TensorBoard
        running_loss += loss.item()
        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, mae_errors=mae_errors)
                      , flush=True)
                writer.add_scalar('training loss',
                                running_loss / args.print_freq,
                                epoch * len(train_loader) + i)
                running_loss = 0.0
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, accu=accuracies,
                       prec=precisions, recall=recalls, f1=fscores,
                       auc=auc_scores)
                      , flush=True)
    

def validate(val_loader, model, criterion, epoch, normalizer, writer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        running_loss = 0.0
        for i, (features, target, batch_cif_ids) in enumerate(val_loader):
            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            
            # compute output
            output = model(features[0], features[1], features[2], features[3])
            loss = criterion(output, target_normed)
    
            # measure accuracy and record loss
            if args.task == 'regression':
                mae_error = mae(normalizer.denorm(output), target)
                losses.update(loss.item(), target.size(0))
                mae_errors.update(mae_error.item(), target.size(0))
                if test:
                    test_pred = normalizer.denorm(output)
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
            else:
                accuracy, precision, recall, fscore, auc_score =\
                    class_eval(output, target)
                losses.update(loss.item(), target.size(0))
                accuracies.update(accuracy.item(), target.size(0))
                precisions.update(precision.item(), target.size(0))
                recalls.update(recall.item(), target.size(0))
                fscores.update(fscore.item(), target.size(0))
                auc_scores.update(auc_score.item(), target.size(0))
                if test:
                    test_pred = torch.exp(output)
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            # write to TensorBoard
            running_loss += loss.item()
            if i % args.print_freq == 0:
                if args.task == 'regression':
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           mae_errors=mae_errors), flush=True)
                    writer.add_scalar('validation loss',
                                    running_loss / args.print_freq,
                                    epoch * len(val_loader) + i)
                    running_loss = 0.0
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           accu=accuracies, prec=precisions, recall=recalls,
                           f1=fscores, auc=auc_scores), flush=True)
 
    if args.task == 'regression':
        print(' * MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors), flush=True)
        return mae_errors.avg
    else:
        print(' * AUC {auc.avg:.3f}'.format(auc=auc_scores), flush=True)
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.detach().numpy())
    target = target.detach().numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, target, is_best):
    out_root = './checkpoints/'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    out_dir = out_root + target
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    torch.save(state, out_dir+'/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(out_dir+'/checkpoint.pth.tar', out_dir+'/model_best.pth.tar')


if __name__ == '__main__':
    main()


