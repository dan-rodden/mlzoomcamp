#!/usr/bin/env python
# coding: utf-8

"""
Training script for XGBoost cancer classification model
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score

# ================================
# Output File Version
# ================================
version = 1.0

# ================================
# User's original helper function
# ================================

def load_pred_df(url: str, sep='\t') -> pd.DataFrame:
    df = pd.read_csv(url, sep=sep)
    df = df.transpose() 

    # cleanup
    df.columns = df.iloc[0, :]
    df = df.drop(['Unnamed: 0'])

    return df


# ================================
# Main Training Script
# ================================

print("Loading gene expression data...")
counts_df = load_pred_df("data/gse68086/GSE68086_TEP_data_matrix.txt/GSE68086_TEP_data_matrix.txt")
counts_df = counts_df.astype('int')

print("Loading metadata...")
meta_data = pd.read_csv("data/gse68086/GSE68086_series_matrix.csv")
meta_data.columns = meta_data.columns.str.strip('!').str.lower().str.replace('.', '_')

# clean up the metadata of double-quotes
obj_cols = list(meta_data.columns[meta_data.dtypes == "object"])

for col in obj_cols:
    meta_data[col] = meta_data[col].str.strip('"')

# create target variable
meta_data['target'] = meta_data['sample_characteristics_ch1_3'].apply(lambda x: 1 if x == 'cancer type: HC' else 0)

# save this metadata for later
target_meta = meta_data.loc[:, ['sample_source_name_ch1', 'target']]

# Filter low expression genes
print("Filtering genes with low expression...")
colsums = pd.Series(counts_df.sum(axis=0))
counts_df = counts_df.loc[:, colsums.values >= 10]

# perform a join on the data values
df = pd.merge(left=target_meta, right=counts_df, how="left", left_on='sample_source_name_ch1', right_on=counts_df.index)

# Train/test split
print("Splitting data into train and test sets...")
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.target.values, shuffle=True, random_state=1)

print(f"Train samples: {len(df_train)}")
print(f"Test samples: {len(df_test)}")

# create the response
y_train = df_train.target.astype(int).values
y_test = df_test.target.astype(int).values

# delete the response from the dataframe along with sample_source_name_ch1
del df_train['sample_source_name_ch1']
del df_train['target']

del df_test['sample_source_name_ch1']
del df_test['target']

# convert to dictionaries
train_dict = df_train.to_dict(orient='records')
test_dict = df_test.to_dict(orient='records')

# vectorize features
print("Vectorizing features...")
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dict)
X_test = dv.transform(test_dict)

# Create DMatrix for XGBoost
print("Training XGBoost model...")
features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# XGBoost parameters (optimized from hyperparameter tuning)
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 6,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

# Train final model
watchlist = [(dtrain, 'train'), (dtest, 'test')]

model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=175,
    verbose_eval=25,
    evals=watchlist
)

# Evaluate model
print("\nEvaluating model on test set...")
y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)

print(f"Test ROC-AUC Score: {auc:.4f}")

# Save the model
print("\nSaving model...")

eta, depth, min_child = xgb_params['eta'], xgb_params['max_depth'], xgb_params['min_child_weight']

output_file = f"xgb_model_eta={eta}_depth={depth}_min-child={min_child}_v{version}.bin"

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"Model saved to {output_file}")
print("Training completed successfully!")