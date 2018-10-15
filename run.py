import pandas as pd
import numpy as np
import sklearn

clickbait_percentage = pd.read_csv('percents.csv')
dic_source_clickbait = {row['source']: row['percent_clickbait'] for i, row in clickbait_percentage.iterrows()}
average_clickbait = clickbait_percentage.percent_clickbait.sum() / len(clickbait_percentage)
data = pd.read_csv('edata_all.csv')

def add_clickbait(source):
    if source.startswith('www.'):
        source = source[4:]
    if source in dic_source_clickbait:
        return dic_source_clickbait[source]
    return average_clickbait
    
data['clickbait_percentage'] = data['source'].apply(add_clickbait)


def get_features(data, source_len = 724):
    """
    features for claims
    """
    dic_f = {} # claimCount -> features
    
    for i in range(len(data)):
        row = data.iloc[i]
        stance = row['articleHeadlineStance']
        stance_id = -1 if stance == 'against' else 0 if stance == 'observing'\
            else 1
        source = row.sourceCount - 1 # 1-index to 0-index
        claim = row.claimCount
        
        if claim not in dic_f: dic_f[claim] = np.zeros((source_len,))
        dic_f[claim][source] = stance_id
    
    #claims = dic_f.keys()
    return dic_f


def extract_truth_labels(data):
    claims = sorted(data.claimCount.unique().tolist())
    l = [''] * len(claims)
    for i in range(len(data)):
        row = data.iloc[i]
        truth = row.claimTruth
        claim = row.claimCount
        claimIdx = claims.index(claim)
        l[claimIdx] = truth        
    return (claims, l)


def build_veracity_prediction_matrix():
    dic_f = get_features(data)
        
    (claims, veracity) = extract_truth_labels(data)
    
    n = len(claims)
    m = dic_f.items()[0][1].shape[0]
    
    F = np.zeros((n, m))
    for i, c in enumerate(claims): F[i, :] = dic_f[c]
    
    return (claims, F, veracity)


claims, F, vera = build_veracity_prediction_matrix()
clf = sklearn.linear_model.LogisticRegression()
# This is the average accuracy for the original matrix (using cross validation)
np.mean(sklearn.model_selection.cross_val_score(clf, F, vera, cv=8))

# now slap the percentage of non-clickbait into the feature matrix
G = F.copy()
for i, row in data.iterrows():
    source_index = row['sourceCount'] - 1
    percent_clickbait = row['clickbait_percentage']
    G[:, source_index] = F[:, source_index] * (1 - percent_clickbait * 0.01)
    

clf_g = sklearn.linear_model.LogisticRegression()
# This is the average accuracy for the matrix with clickbait slapped
np.mean(sklearn.model_selection.cross_val_score(clf_g, G, vera, cv=8))
