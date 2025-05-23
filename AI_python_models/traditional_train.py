import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from joblib import dump
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition   import TruncatedSVD
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv('liar2_preprocessed.csv')
X = df.drop(columns=['label'])
Y = df['label']                               

# Split estratificado train / temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=200900
)

# Split estratificado val / test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=200900
)

extra_stop = {"say","said","state","people","one","rate","year","percent","poster"}

combined_stop = list(ENGLISH_STOP_WORDS.union(extra_stop))

# ----------------------------------- -----------------------------------
#
#                       TRANSFORMADORES DE TEXTO
#
# ----------------------------------- -----------------------------------
# Capturamos ngramas de palabras y ngramas de caracteres para capturar sufijos y prefijos
word_vect = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    min_df=5, 
    max_df=0.9,
    stop_words=combined_stop,
    token_pattern=r"(?u)\b\w\w+\b"
)
char_vect = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    min_df=5
)

tfidf_text_features = FeatureUnion([
    ('tfidf_word', word_vect),
    ('tfidf_char', char_vect)
])

# Usando n gramas 1,1 en caso de palabras para subject
word_vect_base = TfidfVectorizer(
    analyzer='word',
    min_df=5, 
    max_df=0.9,
    stop_words=combined_stop,
    token_pattern=r"(?u)\b\w\w+\b"
)
char_vect_base = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    min_df=5
)

tfidf_base_text_features = FeatureUnion([
    ('tfidf_word', word_vect_base),
    ('tfidf_char', char_vect_base)
])


# ----------------------------------- -----------------------------------
#
#                                PIPELINES
#
# ----------------------------------- -----------------------------------

base_tf_idf =  Pipeline([('tfidf', tfidf_base_text_features)])

tfidf_pipeline = Pipeline([
    ('tfidf', tfidf_text_features)
])

tfidf_svd_pipeline = Pipeline([
    ('tfidf', tfidf_text_features),
    ('svd',   TruncatedSVD(n_components=12, random_state=200900))
])

subject_tfidf_svd_pipeline = Pipeline([
    ('tfidf', tfidf_base_text_features),
    ('svd',   TruncatedSVD(n_components=12, random_state=200900))
])

base_bow =  Pipeline([('bow',CountVectorizer(min_df=5, max_df=0.9, stop_words=combined_stop, token_pattern=r"(?u)\b\w\w+\b"))])

bow_pipeline = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9, stop_words=combined_stop, token_pattern=r"(?u)\b\w\w+\b"))
])

num_cols = ['year','month','day','dayofweek','statement_len_chars','statement_len_words',
            'speaker_description_len_chars','speaker_description_len_words','justification_len_chars',
            'justification_len_words','total_counts','true_prop','mostly_true_prop','half_true_prop',
            'mostly_false_prop','false_prop','pants_on_fire_prop']

cat_cols = ['ctx_document',
            'ctx_location',
            'ctx_media',
            'ctx_social_media',
            'ctx_verbal_event']

num_pipeline = Pipeline([('scale', StandardScaler())])
cat_pipeline = Pipeline([('ohe',   OneHotEncoder(handle_unknown='ignore'))])

num_pipeline_range = Pipeline([
    ('scale', MinMaxScaler(feature_range=(0,1)))
])

bool_cat_pipeline = Pipeline([
    ('ord',  OrdinalEncoder(dtype=int)),
])

# TF-IDF sin SVD (para SVM y RegLog)
pre_tfidf = ColumnTransformer([
    ('stmt',  tfidf_pipeline,   'clean_statement'),
    ('spkdesc', tfidf_pipeline, 'clean_speaker_description'),
    ('justf', tfidf_pipeline,   'clean_justification'),
    ('subj',  base_tf_idf,      'subject'),
    ('nums',  num_pipeline,     num_cols),
    ('cats',  cat_pipeline,     cat_cols),
])

# TF-IDF + SVD (RandomForest y GradientBoost)
pre_tfidf_svd = ColumnTransformer([
    ('stmt',    tfidf_svd_pipeline,         'clean_statement'),
    ('spkdesc', tfidf_svd_pipeline,         'clean_speaker_description'),
    ('justf',   tfidf_svd_pipeline,         'clean_justification'),
    ('subj',    subject_tfidf_svd_pipeline, 'subject'),
    ('nums',    num_pipeline,               num_cols),
    ('cats',    cat_pipeline,               cat_cols),
])

# Bag of words (Naive Bayes)
pre_bow = ColumnTransformer([
    ('stmt',    bow_pipeline,       'clean_statement'),
    ('spkdesc', bow_pipeline,       'clean_speaker_description'),
    ('justf',   bow_pipeline,       'clean_justification'),
    ('subj',    base_bow,           'subject'),
    ('nums',    num_pipeline_range, num_cols),
    ('cats',    bool_cat_pipeline,  cat_cols),
])


# Define los modelos
model_pipelines = {
    'SVM'    : ImbPipeline([('pre', pre_tfidf),     ('smote', SMOTE(random_state=200900)),   ('clf', LinearSVC(class_weight='balanced', max_iter=5000, random_state=200900))]),
    'LogReg' : ImbPipeline([('pre', pre_tfidf),     ('smote', SMOTE(random_state=200900)),   ('clf', LogisticRegression(class_weight='balanced', solver='saga', max_iter=2000, random_state=200900))]),
    'RF'     : ImbPipeline([('pre', pre_tfidf_svd), ('smote', SMOTE(random_state=200900)),   ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=200900))]),
    'GB'     : ImbPipeline([('pre', pre_tfidf_svd), ('smote', SMOTE(random_state=200900)),   ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=200900))]),
    'NB'     : ImbPipeline([('pre', pre_bow),       ('smote', SMOTE(random_state=200900)),   ('clf', MultinomialNB())]),
}

# Creación de grid de hiperparámetros para los distintos modelos
param_grids = {
    'SVM':    {'clf__C': [0.01, 0.1, 1, 10]},
    'LogReg': {'clf__C': [0.01, 0.1, 1, 10]},
    'RF':     {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 10, 20]},
    'GB':     {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.01, 0.1]},
    'NB':     {'clf__alpha': [0.5, 1.0]}
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