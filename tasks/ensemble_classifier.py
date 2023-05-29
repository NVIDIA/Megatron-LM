import os
import argparse
from collections import defaultdict

import numpy as np
import torch


def process_files(args):
    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)
    all_uid = defaultdict(list)
    
    for path in args.paths:
        path = os.path.join(path, args.prediction_name)
        try:
            data = torch.load(path)
            for dataset in data:
                name, d = dataset
                predictions, labels, uid = d
                all_predictions[name].append(np.array(predictions))
                if args.labels is None:
                    args.labels = [i for i in range(predictions.shape[1])]
                if args.eval:
                    all_labels[name].append(np.array(labels))
                all_uid[name].append(np.array(uid))
        except Exception as e:
            print(e)
            continue
    
    return all_predictions, all_labels, all_uid


def get_threshold(all_predictions, all_labels, one_threshold=False):
    if one_threshold:
        combined_preds = np.concatenate(list(all_predictions.values()))
        combined_labels = np.concatenate(list(all_labels.values()))
        all_predictions = {'combined': combined_preds}
        all_labels = {'combined': combined_labels}
    
    out_thresh = []
    for dataset in all_predictions:
        preds = np.concatenate(all_predictions[dataset])
        labels = np.concatenate(all_labels[dataset])
        out_thresh.append(calc_threshold(preds, labels))
    
    return out_thresh


def calc_threshold(p, l):
    trials = np.arange(0, 1, 0.01)
    best_acc = float('-inf')
    best_thresh = 0
    
    for t in trials:
        acc = ((apply_threshold(p, t).argmax(-1) == l).astype(float)).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    
    return best_thresh


def apply_threshold(preds, t):
    prob = preds[:, -1]
    thresholded = (prob >= t).astype(int)
    preds = np.zeros_like(preds)
    preds[np.arange(len(thresholded)), thresholded.reshape(-1)] = 1
    return preds


def threshold_predictions(all_predictions, threshold):
    if len(threshold) != len(all_predictions):
        threshold = [threshold[-1]] * (len(all_predictions) - len(threshold))
    
    for i, dataset in enumerate(all_predictions):
        thresh = threshold[i]
        preds = np.concatenate(all_predictions[dataset])
        all_predictions[dataset] = apply_threshold(preds, thresh)
    
    return all_predictions


def postprocess_predictions(all_predictions, all_labels, args):
    for d in all_predictions:
        all_predictions[d] = np.mean(all_predictions[d], axis=0)
    
    if args.calc_threshold:
        args.threshold = get_threshold(all_predictions, all_labels, args.one_threshold)
        print('threshold', args.threshold)
    
    if args.threshold is not None:
        all_predictions = threshold_predictions(all_predictions, args.threshold)
    
    return all_predictions, all_labels


def write_predictions(all_predictions, all_labels, all_uid, args):
    all_correct = 0
    count = 0
    
    for dataset in all_predictions:
        preds = np.argmax(all_predictions[dataset], -1)
        
        if args.eval:
            correct = (preds == np.concatenate(all_labels[dataset])).sum()
            num = len(all_labels[dataset])
            accuracy = correct / num
            count += num
            all_correct += correct
            print(accuracy)
        
        dataset_dir = os.path.join(args.outdir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        outpath = os.path.join(dataset_dir, os.path.splitext(args.prediction_name)[0] + '.tsv')
        
        with open(outpath, 'w') as f:
            f.write('id\tlabel\n')
            for uid, p in zip(np.concatenate(all_uid[dataset]), preds.tolist()):
                f.write(f'{uid}\t{args.labels[p]}\n')
    
    if args.eval:
        print(all_correct / count)


def ensemble_predictions(args):
    all_predictions, all_labels, all_uid = process_files(args)
    all_predictions, all_labels = postprocess_predictions(all_predictions, all_labels, args)
    write_predictions(all_predictions, all_labels, all_uid, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', required=True, nargs='+', help='paths to checkpoint directories used in ensemble')
    parser.add_argument('--eval', action='store_true', help='compute accuracy metrics against labels (dev set)')
    parser.add_argument('--outdir', help='directory to place ensembled predictions in')
    parser.add_argument('--prediction-name', default='test_predictions.pt', help='name of predictions in checkpoint directories')
    parser.add_argument('--calc-threshold', action='store_true', help='calculate threshold classification')
    parser.add_argument('--one-threshold', action='store_true', help='use one threshold for all subdatasets')
    parser.add_argument('--threshold', nargs='+', default=None, type=float, help='user supplied threshold for classification')
    parser.add_argument('--labels', nargs='+', default=None, help='whitespace separated list of label names')
    args = parser.parse_args()
    ensemble_predictions(args)


if __name__ == '__main__':
    main()
