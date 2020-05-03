import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

import datetime
import uuid
import tensorboard_logger
from resnet.resnetcifar import *
from densenet.densenetcifar import *
from datasets.noisycifar import NCIFAR10
# from datasets.noisycifar import NCIFAR100
# from datasets.nsvhn import NSVHN

parser = argparse.ArgumentParser(description='PyTorch ResNet and PP-ResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default], cifar100, svhn)')
parser.add_argument('--arch', default='resnet', type=str, help='architecture (resnet, densenet, [... more to come ...])')

parser.add_argument('--epochs', default=160, type=int, help='number of total epochs to run')
parser.add_argument('--milestones', default='[80, 120]', type=str, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--extra-epochs', default=0, type=int, help='number of extra epochs to run')
parser.add_argument('--extra-milestones', default='[160]', type=str, help='extra epoch milestones for the scheduler')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 5e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=20, type=int, help='total number of layers (default: 20)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)') # for densenet
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-efficient', dest='efficient', action='store_false',
                    help='To not use bottleneck block')

parser.add_argument('--pushpull', action='store_true', help='use Push-Pull layer as 1st layer (default: False)')
parser.add_argument('--pp-block1', action='store_true', help='use 1st PushPull residual block')
parser.add_argument('--pp-all', action='store_true', help='use all PushPull residual block')

parser.add_argument('--train-alpha', action='store_true', help='whether to learn the values of alpha ')
parser.add_argument('--alpha-pp', default=1, type=float, help='inhibition factor (default: 1.0)')
parser.add_argument('--scale-pp', default=2, type=float, help='upsampling factor for PP kernels (default: 2)')

parser.add_argument('--lpf-size', default=None, type=int, help='Size of the LPF for anti-aliasing (default: 1)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='resnet20', type=str, help='name of experiment')

parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)

best_prec1 = 0
use_cuda = False


def main():
    global args, best_prec1, use_cuda
    args = parser.parse_args()

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': 0, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn')

    if args.dataset == 'cifar10':
        nclasses = 10
        dataset_train = NCIFAR10('./data', train=True,
                                 transform=transform_train,
                                 normalize_transform=normalize)
        dataset_test = NCIFAR10('./data', train=False,
                                transform=transform_test,
                                normalize_transform=normalize)
    else:
        raise RuntimeError('no other data set implementations available')
    '''
    elif args.dataset == 'cifar100':
        nclasses = 100
        dataset_train = NCIFAR100('./data', train=True, transform=transform_train,
                                  normalize_transform=normalize, download=True)
        dataset_test = NCIFAR100('./data', train=False, transform=transform_test,
                                 normalize_transform=normalize, download=True)
    elif args.dataset == 'svhn':
        nclasses = 10
        dataset_train = NSVHN('./data', split='train', transform=transform_train,
                              normalize_transform=normalize)
        dataset_test = NSVHN('./data', split='test', transform=transform_test,
                             normalize_transform=normalize)
    '''

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    # --------------------------------------------------------------------------------
    # create model
    if args.arch == 'resnet':
        experiment_dir = 'experiments/'
        output_dir = experiment_dir + 'resnet-cifar/'

        rnargs = {'use_pp1': args.pushpull,
                  'pp_block1': args.pp_block1,
                  'pp_all': args.pp_all,
                  'train_alpha': args.train_alpha,
                  'size_lpf': args.lpf_size}

        if args.layers == 20:
            model = resnet20(**rnargs)
        elif args.layers == 32:
            model = resnet32(**rnargs)
        elif args.layers == 44:
            model = resnet44(**rnargs)
        elif args.layers == 56:
            model = resnet56(**rnargs)
    elif args.arch == 'densenet':
        experiment_dir = 'experiments/'
        output_dir = experiment_dir + 'densenet-cifar/'
        rnargs = {'use_pp1': args.pushpull,
                  'pp_block1': args.pp_block1,
                  'num_classes': nclasses,
                  'small_inputs': True,
                  'efficient': args.efficient,
                  'compression': args.reduce,
                  'drop_rate': args.droprate,
                  # 'scale_pp': args.scale_pp,
                  # 'alpha_pp': args.alpha_pp
                  }

        if args.layers == 40:
            model = densenet40_12(**rnargs)
        elif args.layers == 100:
            if args.growth == 12:
                model = densenet100_12(**rnargs)
            elif args.growth == 24:
                model = densenet100_24(**rnargs)
    else:
        raise RuntimeError('chosen architecture not implemented (yet)...')

    logger = None
    if args.tensorboard:
        ustr = datetime.datetime.now().strftime("%y-%m-%d_%H-%M_") + uuid.uuid4().hex[:3]
        logger = tensorboard_logger.Logger(experiment_dir + "tensorboard/" + args.name + '/' + ustr)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # --------------------------------------------------------------------------------

    use_cuda = torch.cuda.is_available()

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    if use_cuda:
        model = model.cuda()

    # optionally resume from a checkpoint
    epoch = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    lr_milestones = json.loads(args.milestones)
    if args.extra_epochs > 0:
        lr_milestones = list(set(lr_milestones + json.loads(args.extra_milestones)))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_milestones,
                                                     gamma=0.1)
    scheduler.step(epoch)

    directory = output_dir + "%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)

    for epoch in range(args.start_epoch, args.epochs + args.extra_epochs):
        fileout = open(output_dir + args.name + '/log.txt', "a+")
        # adjust_learning_rate(logger, optimizer, epoch + 1, args.epochs)
        scheduler.step()
        print('lr(', epoch, '): ', scheduler.get_lr())

        # train for one epoch
        train(logger, train_loader, model, criterion, optimizer, epoch, fileout)

        # evaluate on validation set
        prec1 = validate(logger, val_loader, model, criterion, epoch, fileout)
        fileout.close()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, output_dir)

    print('Best accuracy: ', best_prec1)
    fileout = open(output_dir + args.name + '/log.txt', "a+")
    fileout.write('Best accuracy: {}\n'.format(best_prec1))
    fileout.close()


def train(logger, train_loader, model, criterion, optimizer, epoch, file=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    global use_cuda

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach()
        output = output.detach()

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i,
                                                                  len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses, top1=top1))
            if file is not None:
                file.write('Epoch: [{0}][{1}/{2}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time, loss=losses,
                                                                              top1=top1))

        if logger is not None:
            if i % (args.print_freq / 2) == 0:
                log_alpha_histograms(logger, epoch * len(train_loader) + i, model)
            logger.log_scalar('train_loss', losses.avg, epoch * len(train_loader) + i)
            logger.log_scalar('train_acc', top1.avg, epoch * len(train_loader) + i)


def to_np(x):
    return x.detach().cpu().numpy()


def log_alpha_histograms(logger, step, model):
    mode = 'train'
    # Log histograms of weights and grads.
    for h_name, h in zip(['model'], [model]):
        for tag, value in h.named_parameters():
            if 'alpha' in tag:
                tag = h_name + '/' + tag.replace('.', '/')
                logger.log_histogram(tag, to_np(value), step)
                # False is temporary to avoid this logging to happen
                if value.grad is not None:
                    logger.log_histogram(tag + '/grad', to_np(value.grad), step, bins=np.linspace(-.2, .2, 100))


def validate(logger, val_loader, model, criterion, epoch, file=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader),
                                                                  batch_time=batch_time, loss=losses,
                                                                  top1=top1))
            if file is not None:
                file.write('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'.format(i, len(val_loader),
                                                                              batch_time=batch_time, loss=losses,
                                                                              top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    file.write(' * Prec@1 {top1.avg:.3f} \n'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        logger.log_scalar('val_loss', losses.avg, epoch)
        logger.log_scalar('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = output_dir + args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + "/" + filename
    torch.save(state, filename)
    # if is_best:
    #    shutil.copyfile(filename, 'resnet/runs/%s/' % (args.name) + 'model_best.pth.tar')


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


def adjust_learning_rate(logger, optimizer, epoch, totepochs):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 180th epochs"""
    if args.dataset == 'cifar10' or 'cifar100':
        lr = args.lr * ((0.1 ** int(epoch >= totepochs * 0.50)) * (0.1 ** int(epoch >= totepochs * 0.75)) *
                        (0.1 ** int(epoch >= totepochs * 0.95)))

        # in the case some extra epochs are needed (full PP network case)
        lr = args.lr * (0.2 ** int(epoch >= totepochs * 1.1))
    elif args.dataset == 'svhn':
        lr = args.lr * ((0.1 ** int(epoch >= totepochs * 0.5)) *
                        (0.1 ** int(epoch >= totepochs * 0.75)))

    # log to TensorBoard
    if args.tensorboard:
        logger.log_scalar('learning_rate', lr, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
