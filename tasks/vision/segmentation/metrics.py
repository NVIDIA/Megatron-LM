#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#copyright (c) go-hiroaki & Chokurei
#email: guangmingwu2010@gmail.com 
#       guozhilingty@gmail.com
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6

def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 4-d tensor in [batch_size, channels, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 4-d binarized y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data

def _argmax(y_data, dim):
    """
    args:
        y_data : 4-d tensor in [batch_size, chs, img_rows, img_cols]
        dim : int
    return 3-d [int] y_data
    """
    return torch.argmax(y_data, dim).int()


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : [int] 3-d in [batch_size, img_rows, img_cols]
        y_pred : [int] 3-d in [batch_size, img_rows, img_cols]
    return [float] true_positive
    """
    return torch.sum(y_true * y_pred).float()


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_positive
    """
    return torch.sum((1 - y_true) * y_pred).float()


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] true_negative
    """
    return torch.sum((1 - y_true) * (1 - y_pred)).float()


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_negative
    """
    return torch.sum(y_true * (1 - y_pred)).float()


def _get_weights(y_true, nb_ch):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        nb_ch : int 
    return [float] weights
    """
    batch_size, img_rows, img_cols = y_true.shape
    pixels = batch_size * img_rows * img_cols
    weights = [torch.sum(y_true==ch).item() / pixels for ch in range(nb_ch)]
    return weights


class CFMatrix(object):
    def __init__(self, des=None):
        self.des = des

    def __repr__(self):
        return "ConfusionMatrix"

    def __call__(self, y_pred, y_true, ignore_index, threshold=0.5):

        """
        args:
            y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return confusion matrix
        """
        batch_size, img_rows, img_cols = y_pred.shape
        chs = ignore_index
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = [nb_tp, nb_fp, nb_tn, nb_fn]
            performs = None
        else:
            performs = torch.zeros(chs, 4).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_false_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_false_ch[torch.logical_and((y_true != ch), (y_true != ignore_index))] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = torch.sum(y_false_ch * y_pred_ch).float()
                nb_tn = torch.sum(y_false_ch * (1 - y_pred_ch)).float()
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch), :] = torch.FloatTensor([nb_tp, nb_fp, nb_tn, nb_fn])
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class OAAcc(object):
    def __init__(self, des="Overall Accuracy"):
        self.des = des

    def __repr__(self):
        return "OAcc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (tp+tn)/total
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)

        nb_tp_tn = torch.sum(y_true == y_pred).float()
        mperforms = nb_tp_tn / (batch_size * img_rows * img_cols)
        performs = None
        return mperforms, performs


class Precision(object):
    def __init__(self, des="Precision"):
        self.des = des

    def __repr__(self):
        return "Prec"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fp)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fp + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fp + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Recall(object):
    def __init__(self, des="Recall"):
        self.des = des

    def __repr__(self):
        return "Reca"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fn)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fn + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fn + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class F1Score(object):
    def __init__(self, des="F1Score"):
        self.des = des

    def __repr__(self):
        return "F1Sc"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return 2*precision*recall/(precision+recall)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            _precision = nb_tp / (nb_tp + nb_fp + esp)
            _recall = nb_tp / (nb_tp + nb_fn + esp)
            mperforms = 2 * _precision * _recall / (_precision + _recall + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                _precision = nb_tp / (nb_tp + nb_fp + esp)
                _recall = nb_tp / (nb_tp + nb_fn + esp)
                performs[int(ch)] = 2 * _precision * \
                    _recall / (_precision + _recall + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Kappa(object):
    def __init__(self, des="Kappa"):
        self.des = des

    def __repr__(self):
        return "Kapp"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (Po-Pe)/(1-Pe)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            nb_total = nb_tp + nb_fp + nb_tn + nb_fn
            Po = (nb_tp + nb_tn) / nb_total
            Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn) +
                  (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
            mperforms = (Po - Pe) / (1 - Pe + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_tn = _get_tn(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                nb_total = nb_tp + nb_fp + nb_tn + nb_fn
                Po = (nb_tp + nb_tn) / nb_total
                Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn)
                      + (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
                performs[int(ch)] = (Po - Pe) / (1 - Pe + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Jaccard(object):
    def __init__(self, des="Jaccard"):
        self.des = des

    def __repr__(self):
        return "Jacc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return intersection / (sum-intersection)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            _intersec = torch.sum(y_true * y_pred).float()
            _sum = torch.sum(y_true + y_pred).float()
            mperforms = _intersec / (_sum - _intersec + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                _intersec = torch.sum(y_true_ch * y_pred_ch).float()
                _sum = torch.sum(y_true_ch + y_pred_ch).float()
                performs[int(ch)] = _intersec / (_sum - _intersec + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class MSE(object):
    def __init__(self, des="Mean Square Error"):
        self.des = des

    def __repr__(self):
        return "MSE"

    def __call__(self, y_pred, y_true, dim=1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return mean_squared_error, smaller the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        return torch.mean((y_pred - y_true) ** 2)


class PSNR(object):
    def __init__(self, des="Peak Signal to Noise Ratio"):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim=1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1 / mse)


class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''
    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret


class AE(object):
    """
    Modified from matlab : colorangle.m, MATLAB V2019b
    angle = acos(RGB1' * RGB2 / (norm(RGB1) * norm(RGB2)));
    angle = 180 / pi * angle;
    """
    def __init__(self, des='average Angular Error'):
        self.des = des

    def __repr__(self):
        return "AE"
    
    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        return average AE, smaller the better
        """
        dotP = torch.sum(y_pred * y_true, dim=1)
        Norm_pred = torch.sqrt(torch.sum(y_pred * y_pred, dim=1))
        Norm_true = torch.sqrt(torch.sum(y_true * y_true, dim=1))
        ae = 180 / math.pi * torch.acos(dotP / (Norm_pred * Norm_true + eps))
        return ae.mean(1).mean(1)


if __name__ == "__main__":
    for ch in [3, 1]:
        batch_size, img_row, img_col = 1, 224, 224
        y_true = torch.rand(batch_size, ch, img_row, img_col)
        noise = torch.zeros(y_true.size()).data.normal_(0, std=0.1)
        y_pred = y_true + noise
        for cuda in [False, True]:
            if cuda:
                y_pred = y_pred.cuda()
                y_true = y_true.cuda()

            print('#'*20, 'Cuda : {} ; size : {}'.format(cuda, y_true.size()))
            ########### similarity metrics
            metric = MSE()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))

            metric = PSNR()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))

            metric = SSIM()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))
                  
            metric = LPIPS(cuda)
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))
            
            metric = AE()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))
            
            ########### accuracy metrics
            metric = OAAcc()
            maccu, accu = metric(y_pred, y_true)
            print('mAccu:', maccu, 'Accu', accu)

            metric = Precision()
            mprec, prec = metric(y_pred, y_true)
            print('mPrec:', mprec, 'Prec', prec)

            metric = Recall()
            mreca, reca = metric(y_pred, y_true)
            print('mReca:', mreca, 'Reca', reca)

            metric = F1Score()
            mf1sc, f1sc = metric(y_pred, y_true)
            print('mF1sc:', mf1sc, 'F1sc', f1sc)

            metric = Kappa()
            mkapp, kapp = metric(y_pred, y_true)
            print('mKapp:', mkapp, 'Kapp', kapp)

            metric = Jaccard()
            mjacc, jacc = metric(y_pred, y_true)
            print('mJacc:', mjacc, 'Jacc', jacc)

