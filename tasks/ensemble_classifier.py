import torch
import os
import numpy as np
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--paths', required=True, nargs='+')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--outdir')
parser.add_argument('--prediction-name', default='test_predictions.pt')
parser.add_argument('--calc-threshold', action='store_true')
parser.add_argument('--one-threshold', action='store_true')
parser.add_argument('--threshold', nargs='+', default=None, type=float)
parser.add_argument('--labels',nargs='+', default=None)
args = parser.parse_args()

all_predictions = collections.OrderedDict()
all_labels = collections.OrderedDict()
all_uid = collections.OrderedDict()
for path in args.paths:
    path = os.path.join(path, args.prediction_name)
    try:
        data = torch.load(path)
        for dataset in data:
            name, d = dataset
            predictions, labels, uid = d
            if name not in all_predictions:
                all_predictions[name] = np.array(predictions)
                if args.labels is None:
                    args.labels = [i for i in range(all_predictions[name].shape[1])]
                if args.eval:
                    all_labels[name] = np.array(labels)
                all_uid[name] = np.array(uid)
            else:
                all_predictions[name] += np.array(predictions)
                assert np.allclose(all_uid[name], np.array(uid))
    except Exception as e:
        print(e)
        continue
all_correct = 0
count = 0
def get_threshold(all_predictions, all_labels):
    if args.one_threshold:
        all_predictons = {'combined': np.concatenate(list(all_predictions.values()))}
        all_labels = {'combined': np.concatenate(list(all_predictions.labels()))}
    out_thresh = []
    for dataset in all_predictions:
        preds = all_predictions[dataset]
        labels = all_labels[dataset]
        out_thresh.append(calc_threshold(preds,labels))
    return out_thresh
def calc_threshold(p, l):
    trials = [(i)*(1./100.) for i in range(100)]
    best_acc = float('-inf')
    best_thresh = 0
    for t in trials:
        acc = ((apply_threshold(p, t).argmax(-1) == l).astype(float)).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh

def apply_threshold(preds, t):
    assert (np.allclose(preds.sum(-1), np.ones(preds.shape[0])))
    prob = preds[:,-1]
    thresholded = (prob >= t).astype(int)
    preds = np.zeros_like(preds)
    preds[np.arange(len(thresholded)), thresholded.reshape(-1)] = 1
    return preds

def threshold_predictions(all_predictions, threshold):
    if len(threshold)!=len(all_predictions):
        threshold = [threshold[-1]]*(len(all_predictions)-len(threshold))
    for i, dataset in enumerate(all_predictions):
        thresh = threshold[i]
        preds = all_predictions[dataset]
        all_predictions[dataset] = apply_threshold(preds, thresh)
    return all_predictions

for d in all_predictions:
    all_predictions[d] = all_predictions[d]/len(args.paths)

if args.calc_threshold:
    args.threshold = get_threshold(all_predictions, all_labels)
    print('threshold', args.threshold)

if args.threshold is not None:
    all_predictions = threshold_predictions(all_predictions, args.threshold)

for dataset in all_predictions:
    preds = all_predictions[dataset]
    preds = np.argmax(preds, -1)
    if args.eval:
        correct = (preds == all_labels[dataset]).sum()
        num = len(all_labels[dataset])
        accuracy = correct/num
        count += num
        all_correct += correct
        accuracy = (preds == all_labels[dataset]).mean()
        print(accuracy)
    if not os.path.exists(os.path.join(args.outdir, dataset)):
        os.makedirs(os.path.join(args.outdir, dataset))
    outpath = os.path.join(args.outdir, dataset, os.path.splitext(args.prediction_name)[0]+'.tsv')
    with open(outpath, 'w') as f:
        f.write('id\tlabel\n')
        f.write('\n'.join(str(uid)+'\t'+str(args.labels[p]) for uid, p in zip(all_uid[dataset], preds.tolist())))
if args.eval:
    print(all_correct/count)
