from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition   import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from traditional_text_transformers import get_base_tfidf_text_features, get_tfidf_text_features, combined_stop


# -----------------------------------------------------------------------
#
#                          PIPELINES - V1
#
# -----------------------------------------------------------------------

base_tf_idf =  Pipeline([('tfidf', get_base_tfidf_text_features())])

tfidf_pipeline = Pipeline([
    ('tfidf', get_tfidf_text_features())
])

tfidf_svd_pipeline = Pipeline([
    ('tfidf', get_tfidf_text_features()),
    ('svd',   TruncatedSVD(n_components=12, random_state=200900)) #12 componentes no representan casi nada de la varianza explicada de nuestros
])

subject_tfidf_svd_pipeline = Pipeline([
    ('tfidf', get_base_tfidf_text_features()),
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


# -----------------------------------------------------------------------
#
#                          PIPELINES - V2
#
# -----------------------------------------------------------------------

tfidf_chi_pipeline = Pipeline([
    ('tfidf',  get_tfidf_text_features()),
    ('select', SelectKBest(chi2, k=2000)),
    ('scale',  MaxAbsScaler()),  
])

tfidf_base_chi_pipeline = Pipeline([
    ('tfidf',  get_base_tfidf_text_features()),
    ('select', SelectKBest(chi2, k=2000)),
    ('scale',  MaxAbsScaler()),  
])

# -------------------------------------------------------------
#
# Funciones creadas en la V1 de los pipelines de entrenamiento
#
# -------------------------------------------------------------

def get_pre_tfidf():
    # TF-IDF sin SVD (para SVM y RegLog)
    return ColumnTransformer([
        ('stmt',    tfidf_pipeline,       'clean_statement'),
        ('spkdesc', tfidf_pipeline,       'clean_speaker_description'),
        ('justf',   tfidf_pipeline,       'clean_justification'),
        ('subj',    base_tf_idf,          'subject'),
        ('nums',    num_pipeline,             num_cols),
        ('cats',    cat_pipeline,             cat_cols),
    ])

def get_pre_tfidf_svd():
    # TF-IDF + SVD (RandomForest y GradientBoost)
    return ColumnTransformer([
        ('stmt',    tfidf_svd_pipeline,         'clean_statement'),
        ('spkdesc', tfidf_svd_pipeline,         'clean_speaker_description'),
        ('justf',   tfidf_svd_pipeline,         'clean_justification'),
        ('subj',    subject_tfidf_svd_pipeline,    'subject'),
        ('nums',    num_pipeline,               num_cols),
        ('cats',    cat_pipeline,               cat_cols),
    ])

def get_pre_tfidf_chi():
    # TF-IDF sin SVD (para SVM y RegLog)
    return ColumnTransformer([
        ('stmt',    tfidf_chi_pipeline,       'clean_statement'),
        ('spkdesc', tfidf_chi_pipeline,       'clean_speaker_description'),
        ('justf',   tfidf_chi_pipeline,       'clean_justification'),
        ('subj',    tfidf_base_chi_pipeline,  'subject'),
        ('nums',    num_pipeline,             num_cols),
        ('cats',    cat_pipeline,             cat_cols),
    ])

def get_pre_bow():
    # Bag of words (Naive Bayes)
    return ColumnTransformer([
        ('stmt',    bow_pipeline,       'clean_statement'),
        ('spkdesc', bow_pipeline,       'clean_speaker_description'),
        ('justf',   bow_pipeline,       'clean_justification'),
        ('subj',    base_bow,           'subject'),
        ('nums',    num_pipeline_range, num_cols),
        ('cats',    bool_cat_pipeline,  cat_cols),
    ])


def to_dense_array(X):
    """Convierte sparse → dense; deja pasar los arrays densos."""
    if sparse.issparse(X):
        return X.toarray()
    return X

# y luego en tu pipeline:
to_dense = FunctionTransformer(
    func=to_dense_array,  # función nombrada, picklable
    accept_sparse=True
)