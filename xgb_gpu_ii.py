def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.exceptions import DataConversionWarning

import numpy as np
import scipy.stats as st
import xgboost as xgb
from prince import MCA
from sklearn.model_selection import RandomizedSearchCV as RSC
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import f1_score as F1
import pandas as pd
import time
start = time.time()
train = pd.read_csv('train.csv')
train['x'] = train['x_exit'].apply(lambda x: 1 if 3750901.5068 <= x <= 3770901.5068 else 0)
train['y'] = train['y_exit'].apply(lambda x: 1 if -19268905.6133 <= x <= -19208905.6133 else 0)
train['target'] = train['x'] + train['y']
train['target'] = train['target'].apply(lambda x: 1 if x == 2 else 0)
train = train.drop(columns=['trajectory_id', 'x', 'y', 'x_exit', 'y_exit'])

test = pd.read_csv('test.csv')
testID = test.trajectory_id
test = test.drop(columns=['trajectory_id', 'x_exit', 'y_exit'])

finaltrain = train.sample(frac = 0.1)
train = train.loc[~train.index.isin(finaltrain.index)]
train = train.reset_index(drop = True)
finaltrain = finaltrain.reset_index(drop = True)
finaltest = finaltrain['target']
finaltrain = finaltrain.drop(columns = ['target'])

y = train['target']
x = train.drop(columns = ['target'])

col = ['highway', 'x_entry', 'y_entry', 'time_gap', 'center_angle', 'prev_state', 'incity_entry', 'entry_hour', 'pace', 'traffic', 'para_angle', 'weather', 'temp']

train = x[col]
finaltrain = finaltrain[col]
test = test[col]

xtrain, xval, ytrain, yval = TTS(train, y, test_size = 0.15, stratify = y)

parameter = {'estimator__n_estimators': st.randint (333, 999), 'estimator__learning_rate': st.uniform (0.01, 0.03),
                    'estimator__gamma': st.uniform (0.3, 1.50), 'estimator__reg_alpha':st.uniform (0.3, 1.50),
                    'estimator__max_depth': st.randint (3, 9), 'estimator__min_child_weight': st.randint (3, 9),
                    'estimator__subsample': [0.5, 0.6, 0.7, 0.8], 'estimator__colsample_bytree': [0.5, 0.6, 0.7, 0.8], 
                    'estimator__tree_method': ['gpu_hist'], 'estimator__early_stopping_rounds' : [20],
                    'estimator__scale_pos_weight' : st.uniform (3, 20), 'estimator__predictor' : ['gpu_predictor']}

model = RSC(xgb.XGBClassifier(), parameter, scoring = 'f1', cv = 3, n_iter = 8)

model.fit(xtrain, ytrain.values.ravel())

def bias_parameter(X, actual):
    result = []
    for _ in list(np.arange(0.49, 0.59, 0.005)):
        target = [1 if x >= _ else 0 for x in X]
        prediction = F1(actual, target)
        result.append([round(prediction, 4), round(_, 3)])

    return max(result)

y_val_pred = model.predict(xval)
y_val_pred = pd.DataFrame(data = y_val_pred, columns = ['target'])

final_pred = model.predict(finaltrain)
final_pred = pd.DataFrame(data = final_pred, columns = ['target'])

print(f'Validation F1: {F1(yval, y_val_pred)}')
print(f'Final Pred F1: {F1(final_pred, finaltest)}')
#print(f'Final Pred F1: {bias_parameter(final_pred.target_1, finaltest)}')

prediction = model.predict(test)
subs = pd.DataFrame({'id': testID})
subx = pd.DataFrame(data = prediction, columns = ['target'])
sub = pd.concat([subs, subx], axis = 1, sort = True)
sub.to_csv('XGB_X.csv', index = False)
print(sum(sub.target))

parameter = []
for x, y in model.best_params_.items():
	new = ''.join(" '{}' : {}".format(x.rpartition('__')[-1], y))
	parameter.append(new)
parameter = ', '.join(parameter)
print(parameter)
end = time.time()
print(end-start)
