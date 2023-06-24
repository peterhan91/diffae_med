import operator
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

def ci_compute(y_true, y_pred, foo, n_iterations=1000, return_data=False):
    # configure bootstrap
    n_iterations = n_iterations
    n_size = int(len(y_true) * 0.50)
    # run bootstrap
    stats = list()
    df = pd.DataFrame(data={'y_pred': y_pred, 'y_true':y_true})
    for i in range(n_iterations):
        # resample
        df_re = df.sample(n=n_size)
        # evaluate model
        score = foo(df_re['y_true'], df_re['y_pred'])
        stats.append(score)
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    if return_data:
        return (lower, upper, np.mean(stats), stats)
    else:
        return (lower, upper, np.mean(stats))


def determine_cutoffs(true, pred, cutoff_num=1000):
    cutoffs = np.linspace(0.0, 1.0, num=cutoff_num)    
    pred, true = np.array(pred), np.array(true)
    ss_result = []
    for cutoff in cutoffs:
        pred_binary = (pred > cutoff).astype(int)
        confusion = confusion_matrix(true, pred_binary)
        TN, FP = confusion[0, 0], confusion[0, 1]
        FN, TP = confusion[1, 0], confusion[1, 1]
        spec = TN / (TN + FP)
        sens = TP / (TP + FN)
        result = (1-spec)*(1-spec) + (1-sens)*(1-sens)
        ss_result.append(result)
    index, _ = min(enumerate(ss_result), key=operator.itemgetter(1))
    return cutoffs[index]

def compute_sensitivity(y_true, y_pred_binary):
    confusion = confusion_matrix(y_true, y_pred_binary)
    FN, TP = confusion[1, 0], confusion[1, 1]
    sens = TP / (TP + FN)
    return sens

def compute_specificity(y_true, y_pred_binary):
    confusion = confusion_matrix(y_true, y_pred_binary)
    TN, FP = confusion[0, 0], confusion[0, 1]
    spec = TN / (TN + FP)
    return spec


if __name__ == '__main__':
    df = pd.read_hdf('../checkpoints/padchest256_autoenc_cls/bootstrap_auc.h5')
    df = df[df['# positive'] > 15].reset_index(drop=True)

    cutoffs = []
    aucs, aucs_lower, aucs_upper = [], [], []
    sens, sens_lower, sens_upper = [], [], []
    spes, spes_lower, spes_upper = [], [], []
    f1s, f1s_lower, f1s_upper = [], [], []

    for patho in tqdm(df['pathology']):
        dfs = df[df['pathology'] == patho]
        y_pred = dfs['y_pred'].tolist()[0]
        y_true = dfs['y_true'].tolist()[0]

        cutoff = determine_cutoffs(y_true, y_pred)
        cutoffs.append(cutoff)
        y_pred_binary = (y_pred > cutoff).astype(int)
        auc, auc_lower, auc_upper = ci_compute(y_true, y_pred, roc_auc_score)
        sen, sen_lower, sen_upper = ci_compute(y_true, y_pred_binary, compute_sensitivity)
        spe, spe_lower, spe_upper = ci_compute(y_true, y_pred_binary, compute_specificity)
        f1, f1_lower, f1_upper = ci_compute(y_true, y_pred_binary, f1_score)
        aucs.append(auc)
        aucs_lower.append(auc_lower)
        aucs_upper.append(auc_upper)
        sens.append(sen)
        sens_lower.append(sen_lower)
        sens_upper.append(sen_upper)
        spes.append(spe)
        spes_lower.append(spe_lower)
        spes_upper.append(spe_upper)
        f1s.append(f1)
        f1s_lower.append(f1_lower)
        f1s_upper.append(f1_upper)
        

    df['cutoff'] = cutoffs
    df['auc'] = aucs   
    df['auc_lower'] = aucs_lower
    df['auc_upper'] = aucs_upper
    df['sensitivity'] = sens
    df['sensitivity_lower'] = sens_lower
    df['sensitivity_upper'] = sens_upper
    df['specificity'] = spes
    df['specificity_lower'] = spes_lower
    df['specificity_upper'] = spes_upper
    df['f1'] = f1s
    df['f1_lower'] = f1s_lower
    df['f1_upper'] = f1s_upper

    df.to_hdf('bootstrap_all.h5', key='df', index=False)


