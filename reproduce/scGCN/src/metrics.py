import numpy as np
import os

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score


def normalised_average_precision(y_true, y_pred):

    from sklearn.metrics.ranking import _binary_clf_curve

    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=None,
                                             sample_weight=None)

    n_pos = np.array(y_true).sum()
    n_neg = (1 - np.array(y_true)).sum()

    precision = tps * n_pos / (tps * n_pos + fps * n_neg)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision, recall, thresholds = np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def closed_set_acc(preds, labels):

    acc = accuracy_score(labels, preds)

    print('close_acc= {:.4f}'.format(acc))

    return acc


def acc_at_95_tpr(open_set_preds, open_set_labels, thresholds, tpr):

    # Error rate at 95% TAR
    _, idx = find_nearest(tpr, 0.95)
    t = thresholds[idx]
    acc_at_95 = acc_at_t(open_set_preds, open_set_labels, t)
    print(f'Error Rate at TPR 95%: {1 - acc_at_95}')

    return acc_at_95


def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print('AUROC= {:.4f}'.format(auroc))

    return auroc


def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):

    if normalised_ap:
        aupr = normalised_average_precision(open_set_labels, open_set_preds)
    else:
        aupr = average_precision_score(open_set_labels, open_set_preds)
    print('AUPR= {:.4f}'.format(aupr))

    return aupr


def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True  Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    print('OSCR= {:.4f}'.format(OSCR))

    return OSCR

'''
    kn_data: closed_set data set
    unk_data: open_set data set
'''

def osr_evaluator(kn_data_closed_pr, kn_data_closed_gt, kn_data_open_pr, unk_data_open_pr=None):
    n_kn = len(kn_data_closed_pr)

    # ===================
    # closed-set eval
    # ===================
    kn_data_acc = closed_set_acc(kn_data_closed_pr, kn_data_closed_gt)

    auroc, aupr, oscr = -1, -1, -1
    if (unk_data_open_pr is not None) and len(unk_data_open_pr) >= 1:
        # ===================
        # open-set eval
        # ===================
        n_unk = len(unk_data_open_pr)

        open_set_pred = list(kn_data_open_pr) + list(unk_data_open_pr)
        open_set_gt   = list(np.zeros(n_kn)) + list(np.ones(n_unk))

        open_set_pred = np.array(open_set_pred)
        open_set_gt   = np.array(open_set_gt)

        auroc = compute_auroc(open_set_pred, open_set_gt)
        aupr  = compute_aupr(open_set_pred,  open_set_gt, normalised_ap=False)

        # OSCR calcs
        open_set_preds_known_cls = open_set_pred[~open_set_gt.astype('bool')]
        open_set_preds_unknown_cls = open_set_pred[open_set_gt.astype('bool')]
        closed_set_preds_pred_cls = kn_data_closed_pr
        labels_known_cls = np.array(kn_data_closed_gt)

        oscr = compute_oscr(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)

    # print('Closed_acc={:.4f}, Open_AUC={:.4f}, Open_AUPR={:.4f}, OSCR={:.4f}'.format(kn_data_acc, auroc, aupr, oscr))

    return kn_data_acc, auroc, aupr, oscr
