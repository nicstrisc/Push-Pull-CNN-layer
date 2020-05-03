import argparse
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import datetime
import uuid
import tensorboard_logger
from resnet.resnetcifar import *
from densenet.densenetcifar import *

from datasets.noisycifar import NCIFAR10
from datasets.noisycifar import NCIFAR100

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--arch', default='resnet', type=str, help='architecture (resnet, densenet, [... more to come ...])')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=20, type=int, help='total number of layers (default: 28)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')  # for densenet
parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-efficient', dest='efficient', action='store_false', help='To not use bottleneck block')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')

parser.add_argument('--pushpull', action='store_true', help='use Push-Pull layer at the beginning (default: False)')
parser.add_argument('--pp-block1', action='store_true', help='use 1st PushPull residual block')
parser.add_argument('--alpha-pp', default=1, type=float, help='inhibition factor (default: 1.0)')
parser.add_argument('--scale-pp', default=2, type=float, help='upsampling factor for PP kernels (default: 2)')
parser.add_argument('--lpf-size', default=None, type=int, help='Size of the LPF for anti-aliasing (default: 1)')

parser.add_argument('--name', default='model_name', type=str, help='name of experiment')

# add noise on the fly (not for CIFAR-C/P)
parser.add_argument('--noise-type', default=None, type=str, help='(default: None) - gaussian, speckle, poisson')
parser.add_argument('--noise-val', default=0, type=float,
                    help='noise parameter: gaussian, speckle -> variance, poisson -> not required (default: 0)')

parser.add_argument('--modelfile', default='checkpoint', type=str, help='name of the file of the model')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--adv-eps', default=0, type=float, help='Epsilon weight for adversarial noise (default: 0)')

parser.set_defaults(augment=True)

best_prec1 = 0
use_cuda = False


def main():
    global args, best_prec1, use_cuda
    args = parser.parse_args()

    adversarial = False
    if args.adv_eps != 0:
        adversarial = True

    use_cuda = torch.cuda.is_available()

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_test = transforms.Compose([
        transforms.ToTensor()  # , normalize
    ])

    kwargs = {'num_workers': 0, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')

    # add noise
    noise = None
    if args.noise_type is not None:
        noise = {'type': args.noise_type, 'val': args.noise_val}

    # data set
    nclasses = 10
    if args.dataset == 'cifar10':
        dataset = NCIFAR10('./data', train=False, transform=transform_test,
                           normalize_transform=normalize, noise_test=noise)
        nclasses = 10
    elif args.dataset == 'cifar100':
        dataset = NCIFAR100('./data', train=False, transform=transform_test,
                            normalize_transform=normalize, noise_test=noise)
        nclasses = 100

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    expdir = ''
    if args.arch == 'resnet':
        expdir = 'models/resnet-cifar/'
        rnargs = {'use_pp1': args.pushpull,
                  'pp_block1': args.pp_block1,
                  # 'pp_all': args.pp_all,
                  # 'train_alpha': args.train_alpha,
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
        expdir = 'models/densenet-cifar/'
        rnargs = {'use_pp1': args.pushpull,
                  'pp_block1': args.pp_block1,
                  'num_classes': nclasses,
                  'small_inputs': True,
                  'efficient': args.efficient,
                  'compression': args.reduce,
                  'drop_rate': args.droprate,
                  'scale_pp': args.scale_pp,
                  'alpha_pp': args.alpha_pp
                  }

        if args.layers == 40:
            model = densenet40_12(**rnargs)
        elif args.layers == 100:
            if args.growth == 12:
                model = densenet100_12(**rnargs)
            elif args.growth == 24:
                model = densenet100_24(**rnargs)
    elif args.arch == 'alexnet':
        expdir = 'models/alexnet-cifar/'
        # model = alexnet.AlexNet(num_classes=nclasses)
    else:
        raise RuntimeError('Fatal error - no other networks implemented')

    # load trained parameters in the model
    if use_cuda:
        trained_model = torch.load(expdir + '%s/' % args.name + args.modelfile + '.pth.tar')
    else:
        trained_model = torch.load(expdir + '%s/' % args.name + args.modelfile + '.pth.tar',
                                   map_location=lambda storage, loc: storage)

    # ------------------ Start loading model ---------------
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    trained_model['state_dict'] = {k: v for k, v in trained_model['state_dict'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(trained_model['state_dict'])
    model.load_state_dict(trained_model['state_dict'])
    # ------------------ Finish loading model --------------

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()  # define loss function (criterion) and optimizer

    # evaluate on validation set
    if args.noise_type is not None:
        fileout = open(expdir + args.name + '/' + noise['type'] + '_' + str(noise['val']) + '.txt', "a+")
    elif adversarial:
        fileout = open(expdir + args.name + '/test_fsgm_' + str(args.adv_eps) + '.txt', "a+")
    else:
        fileout = open(expdir + args.name + '/test_clean.txt', "a+")

    prec1 = validate(val_loader, model, criterion, adversarial_eps=args.adv_eps, file=fileout)
    fileout.write('Test accuracy:\n{}'.format(prec1))
    fileout.close()


def validate(val_loader, model, criterion, adversarial_eps=0, file=None):
    global use_cuda
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    if adversarial_eps == 0:
        model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        # compute output
        if adversarial_eps == 0:
            with torch.no_grad():
                output = model(input_var)
            loss = criterion(output, target_var)
        else:  # Add Adversarial perturbation
            output = model(input_var)
            loss = criterion(output, target_var)
            loss.backward()

            x_grad = torch.sign(input_var.grad.data)
            x_adversarial = torch.clamp(input_var.data + adversarial_eps * x_grad, 0, 1)

            # Classification after optimization
            output = model(torch.autograd.Variable(x_adversarial))
            loss = criterion(output, target_var)

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
    return top1.avg


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
