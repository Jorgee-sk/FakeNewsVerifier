
from joblib import dump
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier 
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from traditional_split_data import data_split
from traditional_pipelines import get_pre_tfidf, get_pre_tfidf_svd, get_pre_bow, get_pre_tfidf_chi, to_dense

X_train, X_val, X_test, y_train, y_val, y_test = data_split()

# Define los modelos
model_pipelines = {
    'SVM'    :    ImbPipeline([('pre', get_pre_tfidf_chi()),     ('smote', SMOTE(random_state=200900)),                             ('clf', LinearSVC(class_weight='balanced', max_iter=20000, random_state=200900, dual=False, tol=1e-4))]),
    'LogReg' :    ImbPipeline([('pre', get_pre_tfidf_chi()),     ('smote', SMOTE(random_state=200900)),                             ('clf', LogisticRegression(class_weight='balanced', solver='saga', max_iter=15000, random_state=200900))]),
    'RF'     :    ImbPipeline([('pre', get_pre_tfidf_chi()),     ('smote', SMOTE(random_state=200900)),                             ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=200900))]),
    'NB'     :    ImbPipeline([('pre', get_pre_bow()),           ('smote', SMOTE(random_state=200900)),                             ('clf', MultinomialNB())]),
    'LogRegCal' : ImbPipeline([('pre', get_pre_tfidf_chi()),     ('smote', SMOTE(random_state=200900)),                             ('clf', CalibratedClassifierCV(estimator=LogisticRegression(solver='saga', class_weight='balanced', max_iter=15000, random_state=200900), method='sigmoid', cv=5))]),
    'HGB'    :    ImbPipeline([('pre', get_pre_tfidf_chi()),     ('smote', SMOTE(random_state=200900)),   ('to_dense', to_dense),   ('clf', HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.1, tol=1e-7, random_state=200900))]),
    'XGBoost':    ImbPipeline([('pre', get_pre_tfidf_chi()),     ('smote', SMOTE(random_state=200900)),                             ('clf', XGBClassifier(eval_metric='mlogloss', random_state=200900))])
}

# Creación de grid de hiperparámetros para los distintos modelos
param_grids = {
    'SVM':    {'clf__C':                [0.01, 0.1, 1, 10]},
    'LogRegCal': {'clf__estimator__C':     [0.01, 0.1, 1, 10]},
    'LogReg': {'clf__C':                  [0.01, 0.1, 1, 10]},
    'RF':     {'clf__n_estimators':       [100, 200], 
               'clf__max_depth':          [None, 10, 20]},
    'NB':     {'clf__alpha':              [0.5, 1.0]},
    'HGB':    {'clf__max_iter':           [300, 400],
               'clf__learning_rate':      [0.01, 0.1],
               'clf__max_depth':          [None, 5, 10]},
    'XGBoost':{
    'clf__n_estimators':      [100,200],
    'clf__learning_rate':     [0.01,0.1],
    'clf__max_depth':         [3,6,9]
    }
}

results = {}

for name, pipeline in model_pipelines.items():
    gs = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=StratifiedKFold(5, shuffle=True, random_state=200900),
        scoring='f1_macro',
        n_jobs=-1,
        error_score='raise'
    )
    gs.fit(X_train, y_train)
    print(f"Mejores params: {gs.best_params_}")
    y_pred = gs.predict(X_val)
    print(f"\n=== {name} best params ===\n", gs.best_params_)
    print(f"\n=== {name} Classification Report on VAL ===")
    print(classification_report(y_val, y_pred, zero_division=0))
    results[name] = gs
    estimator = gs.best_estimator_
    dump(
        estimator,
        f"{name.lower()}_pipeline.joblib"
    )

# Selecciona el mejor y test final
best_name, best_gs = max(results.items(), key=lambda x: x[1].best_score_)
print(f"\n**Mejor modelo: {best_name}**")
y_test_pred = best_gs.predict(X_test)
print("\n=== Test final ===")
print(classification_report(y_test, y_test_pred, zero_division=0))
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, normalize='true')
plt.title("Matriz de confusión normalizada")
plt.show()