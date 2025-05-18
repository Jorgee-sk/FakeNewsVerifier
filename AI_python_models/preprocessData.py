import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset LIAR2 desde la ruta indicada.
    """
    df = pd.read_csv(path)
    return df

def prepare_date_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna 'date' en datetime y extrae características:
      - parseo con formato "%B %d, %Y"
      - year, month, day, dayofweek
    """
    df['date'] = pd.to_datetime(df.get('date', ''), format='%B %d, %Y', errors='coerce')
    df['year'] = df['date'].dt.year.astype(int)
    df['month'] = df['date'].dt.month.astype(int)
    df['day'] = df['date'].dt.day.astype(int)
    df['dayofweek'] = df['date'].dt.dayofweek.astype(int)
    return df

def clean_text_basic(text: str) -> str:
    """
    Limpieza mínima del texto:
      - Minúsculas
      - Eliminación de URLs
      - Eliminación de caracteres no alfanuméricos
      - Eliminación de dígitos
      - Eliminación de espacios
    """
    text = text.lower()                             # Lowercasing
    text = re.sub(r'http\S+|www\.\S+', '', text)    # Eliminar URLs
    text = re.sub(r'[^a-z0-9\s]', '', text)         # Quitar caracteres especiales
    text = re.sub(r'\d+', '', text)                 # Quitar dígitos
    return text.strip()                             # Eliminar espacios


def prepare_text_classic(df_original: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """
    Pipeline clásico para textos:
        Limpieza básica
        Tokenización (NLTK word_tokenize)
        Stop-words
        Lematización
        Reconstrucción del texto limpio
        Cálculo de longitudes
    """
    df = df_original.copy()

    for column_name in column_names:
        processedColumn = 'processed_'+column_name
        tokenColumn = 'tokens_'+column_name
        cleanColumn = 'clean_statement_'+column_name
        lenCharsColumn = column_name+'_len_chars'
        lenWordsColumn = column_name+'_len_words'

        df[processedColumn] = df[column_name].apply(clean_text_basic)     # Limpieza
        df[tokenColumn] = df[processedColumn].apply(word_tokenize)           # Tokenización

        # Stop-words
        sw = set(stopwords.words('english'))
        df[tokenColumn] = df[tokenColumn].apply(lambda toks: [t for t in toks if t not in sw])

        # Lematización
        lemm = WordNetLemmatizer()
        df[tokenColumn] = df[tokenColumn].apply(lambda toks: [lemm.lemmatize(t) for t in toks])

        # Reconstruir y métricas
        df[cleanColumn] = df[tokenColumn].apply(lambda toks: ' '.join(toks))
        df[lenCharsColumn] = df[cleanColumn].str.len()
        df[lenWordsColumn] = df[cleanColumn].str.split().str.len()

    return df


def prepare_statement_transformer(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline mínimo para Transformers:
        Limpieza básica
        No tokenizar manualmente ni quitar stop-words
        Devolver texto listo para el tokenizador del modelo
    """
    df = df_original.copy()
    df['clean_statement'] = df['statement'].apply(clean_text_basic)

    # Las longitudes pueden aportar información al modelo
    df['stmt_len_chars'] = df['clean_statement'].str.len()
    df['stmt_len_words'] = df['clean_statement'].str.split().str.len()
    return df


# LA VARIABLE DATE SE PUEDE BORRAR  HAY QUE BORRAR PROCESSED Y TOKENS ANTES DE PASARLO AL MODELO SON TEMPORALES

if __name__ == '__main__':
    df = load_data('liar2_no_null_data.csv')

    df = prepare_date_for_model(df)

    df = prepare_text_classic(df, ['statement','justification'])

    df.to_csv('liar2_test.csv',index=False)
