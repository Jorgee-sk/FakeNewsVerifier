from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import FeatureUnion


# ----------------------------------------------------------------------
#
#                       TRANSFORMADORES DE TEXTO
#
# ----------------------------------------------------------------------
# Capturamos ngramas de palabras y ngramas de caracteres para capturar sufijos y prefijos

extra_stop = {"say","said","state","people","one","rate","year","percent","poster"}

combined_stop = list(ENGLISH_STOP_WORDS.union(extra_stop))

def get_tfidf_text_features():

    word_vect = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    min_df=0.001, 
    max_df=0.9,
    stop_words=combined_stop,
    token_pattern=r"(?u)\b\w\w+\b",
    norm=None
    )

    char_vect = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3,5),
        min_df=0.001,
        norm=None
    )

    return FeatureUnion([
        ('tfidf_word', word_vect),
        ('tfidf_char', char_vect)
    ])


def get_base_tfidf_text_features():
    # Usando n gramas 1,1 en caso de palabras para subject
    word_vect_base = TfidfVectorizer(
        analyzer='word',
        min_df=0.001, 
        max_df=0.9,
        stop_words=combined_stop,
        token_pattern=r"(?u)\b\w\w+\b",
        norm=None
    )
    char_vect_base = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3,5),
        min_df=0.001,
        norm=None
    )

    return FeatureUnion([
        ('tfidf_word', word_vect_base),
        ('tfidf_char', char_vect_base)
    ])