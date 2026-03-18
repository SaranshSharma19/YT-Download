import pandas as pd
import numpy as np
import time
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

X_train = np.random.rand(5000, 18)
y_train = np.random.randint(0, 3, 5000)
X_val = np.random.rand(1000, 18)
y_val = np.random.randint(0, 3, 1000)

print("Starting LightGBM")
start = time.time()
model1 = LGBMClassifier(n_estimators=500, learning_rate=0.01, num_leaves=32, subsample=0.8, colsample_bytree=0.8, objective='multiclass', n_jobs=4, random_state=42, verbosity=-1)
model1.fit(X_train, y_train)
print(f"LightGBM took {time.time() - start:.2f}s")

print("Starting XGBoost")
start = time.time()
model2 = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective='multi:softprob', eval_metric='mlogloss', n_jobs=4, random_state=42, verbosity=0)
model2.fit(X_train, y_train)
print(f"XGBoost took {time.time() - start:.2f}s")

print("Starting CatBoost")
start = time.time()
model3 = CatBoostClassifier(iterations=500, learning_rate=0.01, depth=6, loss_function='MultiClass', verbose=False, random_seed=42)
model3.fit(X_train, y_train)
print(f"CatBoost took {time.time() - start:.2f}s")
