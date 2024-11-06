#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrian Yan
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


random.seed(18196483)
data = pd.read_csv('spotify52kData.csv')
data.describe()

#Fill missing numerical data with median
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

#Fill missing categorical columns with mode
cat_cols = data.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

#Return with duplicate rows removed
data = data.drop_duplicates()


#Problem 1: Feature Normal Distribution?
sns.set(style="whitegrid")
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axes = plt.subplots(2, 5, figsize=(20, 10)) 
fig.suptitle('Distributions of Song Features', fontsize=16)

for i, feature in enumerate(features, 1):
    plt.subplot(2, 5, i)
    sns.histplot(data[feature], bins=50)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

normality_results = {}

for feature in features:
    stat, p = shapiro(data[feature])
    normality_results[feature] = {'Statistics': stat, 'p-value': f'{p:.50f}'}

normality_df = pd.DataFrame(normality_results).T


#Problem 2: Song Length and Popularity Relationship?
plt.figure(figsize=(10, 8))
sns.scatterplot(x='duration', y='popularity', data=data)
plt.title('Song Duration vs. Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
correlation = data['duration'].corr(data['popularity'])


#Problem 3: Explicit vs Non-Explicit Songs Popularity?
explicit_data = data[data['explicit'] == True]['popularity']
non_explicit_data = data[data['explicit'] == False]['popularity']
stat, p_value = mannwhitneyu(explicit_data, non_explicit_data, alternative='greater')


#Problem 4: Major Key vs Minor Key Popularity?
major_songs = data[data['mode'] == 1]['popularity']
minor_songs = data[data['mode'] == 0]['popularity']
stat, p_value = mannwhitneyu(major_songs, minor_songs, alternative='greater')


#Problem 5: Energy vs Loudness?
plt.figure(figsize=(10, 8))
sns.scatterplot(x='loudness', y='energy', data=data)
plt.title('Effect of Loudness on Energy of Songs')
plt.xlabel('Loudness (dB)')
plt.ylabel('Energy')
plt.show()
correlation = data['loudness'].corr(data['energy'])


#Problem 6: Which Feature Predicts Popularity Best?
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'popularity'
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=18196483)

results = {}
for feature in features:

    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]

    model = LinearRegression()
    model.fit(X_train_feature, y_train)

    y_pred = model.predict(X_test_feature)
    r_squared = r2_score(y_test, y_pred)
    
    results[feature] = r_squared

results_df = pd.DataFrame(list(results.items()), columns=['Feature', 'R_squared'])


#Problem 7: Model With All Song Features
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18196483)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)


#Problem 8: PCA and Extraction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=len(features))
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(features) + 1), explained_variance, alpha=0.6, align='center', label='Individual explained variance')
plt.step(range(1, len(features) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Principal Component Analysis')



#Problem 9: Major/Minor Prediction Logistic Regression
X = data['valence'].values.reshape(-1, 1)  
y = data['mode'].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18196483)
model = LogisticRegression()
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

X = data['speechiness'].values.reshape(-1, 1)  
y = data['mode'].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18196483)
model = LogisticRegression()
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")


#Problem 10: Predictor of Classical Music
data['is_classical'] = (data['track_genre'] == 'classical').astype(int)
X_duration = data['duration'].values.reshape(-1, 1)
y = data['is_classical'].values
X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(X_duration, y, test_size=0.2, random_state=18196483)
model_duration = LogisticRegression()
model_duration.fit(X_train_dur, y_train_dur)
y_pred_dur = model_duration.predict_proba(X_test_dur)[:, 1]
auc_duration = roc_auc_score(y_test_dur, y_pred_dur)
fpr, tpr, thresholds = roc_curve(y_test_dur, y_pred_dur)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {auc_duration:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve For Duration')
plt.legend(loc="lower right")
plt.show()

pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=18196483)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve For Principal Components')
plt.legend(loc="lower right")
plt.show()

#Extra Credit: Correlation between Time Signature and Energy
energy_by_time_sig = data.groupby('time_signature')['energy'].mean().reset_index()

plt.figure(figsize=(8, 4))
sns.barplot(x='time_signature', y='energy', data=energy_by_time_sig, palette='viridis')
plt.title('Average Energy by Time Signature')
plt.xlabel('Time Signature')
plt.ylabel('Average Energy')
plt.show()
correlation = data['time_signature'].corr(data['energy'])
