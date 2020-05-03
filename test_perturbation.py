import argparse
import time
import numpy as np
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# from alexnet import alexnet
from resnet.resnetcifar import *
from densenet.densenetcifar import *
# from wideresnet.wideresnet import *
from datasets.cifarperturbed import CIFAR10_P
from datasets.cifarperturbed import CIFAR100_P
from datasets.noisycifar import NCIFAR10
from datasets.noisycifar import NCIFAR100

parser = argparse.ArgumentParser(description='Test on CIFAR-10-C')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--perturbed-datadir', default='', type=str, help='root path of the CIFAR-C dataset')

parser.add_argument('-b', '--batch-size', default=4, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=50, type=int, help='print frequency (default: 10)')

parser.add_argument('--layers', default=20, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor (default: 1)')

parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='To not use bottleneck block')
parser.add_argument('--no-efficient', dest='efficient', action='store_false', help='to not use efficient impl.')

parser.add_argument('--pushpull', action='store_true', help='use Push-Pull as 1st layer (default: False)')
parser.add_argument('--pp-block1', action='store_true', help='use 1st PushPull residual block')
parser.add_argument('--pp-block1-reduced', action='store_true', help='use 1st PushPull residual block reduced')
parser.add_argument('--modelfile', default='checkpoint', type=str, help='name of the file of the model')
parser.add_argument('--pp-all', action='store_true', help='use all PushPull residual block')
parser.add_argument('--pp-all-reduced', action='store_true', help='use all PushPull residual block reduced')
parser.add_argument('--alpha-pp', default=1, type=float, help='inhibition factor (default: 1.0)')
parser.add_argument('--scale-pp', default=2, type=float, help='upsampling factor for PP kernels (default: 2)')

parser.add_argument('--train-alpha', action='store_true', help='train alpha of push-pull kernels (Default: False)')

parser.add_argument('--lpf-size', default=None, type=int, help='Size of the LPF for anti-aliasing (default: 1)')

parser.add_argument('--arch', default='resnet', type=str, help='architecture (resnet, densenet, ...)')
parser.add_argument('--name', default='01-20', type=str, help='name of experiment-model')

args = parser.parse_args()

best_prec1 = 0
use_cuda = False

perturbations = ['gaussian_noise', 'shot_noise',
                 'motion_blur', 'zoom_blur',
                 'spatter', 'brightness',
                 'translate', 'rotate', 'tilt', 'scale',
                 'speckle_noise', 'gaussian_blur', 'snow', 'shear']
# perturbations = ['translate']


# Root folder of the CIFAR-C and CIFAR-P data sets
# Please change it with the path to the folder where you un-tar the CIFAR-P data set
#
pert_dataset_root = '/default/path/to/CIFAR-C/root/folder/'
pert_dataset_root = '/home/nicola/Scrivania/RESEARCH/Projects/InhibCNN/data/'


NCLASSES = 10
identity = np.asarray(range(1, NCLASSES + 1))
cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (NCLASSES-1 - 5)))
recip = 1./identity


def dist(sigma, mode='top5'):
    if mode == 'top5':
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        return np.sum(np.abs(recip - recip[sigma-1])*recip)


def ranking_dist(ranks, noise_perturbation=False, mode='top5'):
    result = 0
    # step_size = 1 if noise_perturbation else args.difficulty
    step_size = 1

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(predictions, noise_perturbation=False):
    result = 0
    # step_size = 1 if noise_perturbation else args.difficulty
    step_size = 1

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result


# ------------------------------ MAIN -------------------------------------
def main():
    global args, best_prec1, use_cuda, pert_dataset_root
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    if args.perturbed_datadir != '':
        pert_dataset_root = args.perturbed_datadir

    # Clean Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
    if args.dataset == 'cifar10':
        clean_data = NCIFAR10('./data', train=False, transform=transform_test, normalize_transform=normalize)
        nclasses = 10

    elif args.dataset == 'cifar100':
        clean_data = NCIFAR100('./data', train=False, transform=transform_test, normalize_transform=normalize)
        nclasses = 100

    clean_loader = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    # --------------------------------------------------------------------------------
    # create model
    expdir = ''
    if args.arch == 'resnet':
        expdir = 'models/resnet-cifar/'
        rnargs = {'use_pp1': args.pushpull,
                  'pp_block1': args.pp_block1,
                  # 'pp_all': args.pp_all,
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
    model.eval()
    # ------------------ Finish loading model --------------

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()  # define loss function (criterion) and optimizer

    # evaluate on validation set
    '''
    fileout = open(expdir + args.name + '/test_clean.txt', "a+")
    prec1 = validate(clean_loader, model, criterion, file=fileout)
    print('Test accuracy clean: \n{}'.format(prec1))
    fileout.write('Test accuracy clean: \n{}'.format(prec1))
    fileout.close()
    '''

    # ------------------------------------------------------------------
    #           VALIDATE ON CIFAR-10-P
    # ------------------------------------------------------------------
    flip_rates = []
    top5_rates = []
    zipf_rates = []
    f1 = open(expdir + args.name + '/P_flipprob.txt', "w+")
    f2 = open(expdir + args.name + '/P_top5dist.txt', "w+")
    f3 = open(expdir + args.name + '/P_zipfdist.txt', "w+")
    for perturbation_name in perturbations:
        predictions, ranks = validate_perturbed(perturbation_name, model)
        flipprob = flip_prob(predictions)
        flip_rates.append(flip_prob)

        top5dist = ranking_dist(ranks, mode='top5')
        top5_rates.append(top5dist)

        zipfdist = ranking_dist(ranks, mode='zipf')
        zipf_rates.append(zipfdist)

        f1.write('Perturbation: {:15s}  | Flip Prob: {:.5f}\n'.format(perturbation_name, flipprob))
        f2.write('Perturbation: {:15s}  | Top5 Distance: {:.5f}\n'.format(perturbation_name, top5dist))
        f3.write('Perturbation: {:15s}  | Zipf Distance: {:.5f}\n'.format(perturbation_name, zipfdist))

        print('Perturbation: {:15s}'.format(perturbation_name))
        print('Flipping Prob\t{:.5f}'.format(flipprob))
        print('Top5 Distance\t{:.5f}'.format(top5dist))
        print('Zipf Distance\t{:.5f}'.format(zipfdist))

    f1.close()
    f2.close()
    f3.close()

    print('\nmFR (unnormalized by AlexNet): {:.5f}'.format(np.mean(flip_rates)))
    print('mT5 (unnormalized by AlexNet): {:.5f}'.format(np.mean(top5_rates)))
    print('mZD (unnormalized by AlexNet): {:.5f}'.format(np.mean(zipf_rates)))


def validate_perturbed(perturbation_name, model):
    model.eval()
    global pert_dataset_root, use_cuda
    # Data loading code
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
    ])
    if args.dataset == 'cifar10':
        dataset = CIFAR10_P(pert_dataset_root, transform=transform_test, pert_category=perturbation_name)
    elif args.dataset == 'cifar100':
        dataset = CIFAR100_P(pert_dataset_root, transform=transform_test, pert_category=perturbation_name)

    kwargs = {'num_workers': 0, 'pin_memory': True}
    perturbed_dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    predictions, ranks = [], []
    for batch_idx, (data, target) in enumerate(perturbed_dataset_loader):
        num_vids = data.size(0)
        data = data.view(-1, 3, 32, 32)
        if use_cuda:
            data = data.cuda()

        output = model(data)
        output = output.detach()

        for vid in output.view(num_vids, -1, NCLASSES):
            predictions.append(vid.argmax(1).to('cpu').numpy())
            ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])
    return predictions, ranks


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
