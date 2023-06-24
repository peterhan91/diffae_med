import scipy
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def boostrap(y_true, y_pred, n_bootstraps):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred) // 2)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    mean, ci_lower, ci_upper = mean_confidence_interval(bootstrapped_scores)
    
    return (mean, ci_lower, ci_upper, bootstrapped_scores)


if __name__ == '__main__':
    n_bootstraps = 1000
    # rng_seed = 42  # control reproducibility
    # rng = np.random.RandomState(rng_seed)

    df = pd.read_csv('../labels/uka_chest.csv')
    BasePaths = ['../checkpoints/ukachest256_autoenc_cls/']
    # df = pd.read_csv('../datasets/chexpert/chexpert_test.csv')
    # BasePaths = glob.glob('../checkpoints/chexpert256_autoenc_cl*/')
    for BasePath in BasePaths:
        y_true = np.load(os.path.join(BasePath, 'y_true.npy'))
        y_pred = np.load(os.path.join(BasePath, 'y_pred.npy'))
        PRED_LABEL = df.columns[1:-1]
        # PRED_LABEL = df.columns[1:-1]
        print('Label length: ', len(PRED_LABEL))

        averages, diseases = [], []
        preds, trues = [], []
        for i in range(len(PRED_LABEL)):
            try:
                auc = roc_auc_score(y_true[:,i], y_pred[:,i])
            except:
                auc = None
            if auc is not None:
                averages.append(auc)
                diseases.append(PRED_LABEL[i])
                preds.append(y_pred[:,i])
                trues.append(y_true[:,i])
            
        result_df = pd.DataFrame({'pathology': diseases,
            'roc-auc': averages
            })

        counts = []
        for patho in result_df['pathology']:
            count = df[patho].sum()
            counts.append(count)
        result_df['y_pred'] = preds
        result_df['y_true'] = trues
        result_df['# positive'] = counts

        means = []
        lowers, uppers = [], []
        bootstraps = []

        for n in tqdm(range(len(result_df))):
            dfs = result_df.iloc[n]
            res = boostrap(dfs['y_true'], dfs['y_pred'], n_bootstraps)
            means.append(res[0])
            lowers.append(res[1])
            uppers.append(res[2])
            bootstraps.append(res[3])
        result_df['mean auc'] = means
        result_df['CI lower'] = lowers
        result_df['CI upper'] = uppers
        result_df['bootstrap'] = bootstraps

        result_df.to_hdf(os.path.join(BasePath, 'bootstrap_auc.h5'), key='result_df', index=False)
