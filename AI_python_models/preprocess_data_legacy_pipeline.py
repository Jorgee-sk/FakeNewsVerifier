import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

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
        cleanColumn = 'clean_'+column_name
        lenCharsColumn = column_name+'_len_chars'
        lenWordsColumn = column_name+'_len_words'

        df[processedColumn] = df[column_name].apply(clean_text_basic)     # Limpieza
        df[tokenColumn] = df[processedColumn].apply(word_tokenize)        # Tokenización

        # Stop-words
        sw = set(stopwords.words('english'))
        df[tokenColumn] = df[tokenColumn].apply(lambda toks: [t for t in toks if t not in sw])

        # Lematización
        lemm = WordNetLemmatizer()
        df[tokenColumn] = df[tokenColumn].apply(lambda toks: [lemm.lemmatize(t) for t in toks])

        # Reconstruir y métricas
        df[cleanColumn] = df[tokenColumn].apply(lambda toks: ' '.join(toks))

        if column_name == 'statement':
            df[cleanColumn] = df[cleanColumn].apply(
                lambda txt: ' '.join(txt.split()[:21])[:153]  #p98 values
            )
        elif column_name == 'speaker_description':
            df[cleanColumn] = df[cleanColumn].apply(
                lambda txt: ' '.join(txt.split()[:50])[:389] #p95 values
            )
        elif column_name == 'justification':
            df[cleanColumn] = df[cleanColumn].apply(
                lambda txt: ' '.join(txt.split()[:150])[:1000] # > p99 & < p100
            )


        df[lenCharsColumn] = df[cleanColumn].str.len()
        df[lenWordsColumn] = df[cleanColumn].str.split().str.len()

        df.drop(columns=[column_name, processedColumn, tokenColumn], inplace=True)

    return df

GENERIC_CTX = {
    'social_media':    ['facebook', 'twitter', 'instagram', 'tiktok', 'youtube', 'snapchat', 'social media', 'post', 'tweet', 'tweets','meme',
                       'viral','podcast','internet','website', 'web post', 'website post', 'web', 'site', 'online', 'network','platform'],

    'advertising':     ['ad', 'tv ad', 'attack ad', 'billboard', 'video ad', 'commercial', 'advertisement', 'campaign commercial','campaign'],

    'verbal_event':    ['interview', 'conversation', 'q&a','question','answer','speech', 'address', 'remarks', 'town hall', 'townhall','comments','declaration',
                         'comments on', 'commentary', 'remarks to reporters', 'message','comment','debate', 'primary debate', 'presidential debate', 'discussion',
                         'session','hearing', 'forum', 'committee', 'roundtable', 'conference', 'fundraiser', 'rally', 'meeting', 'presentation','briefing', 
                         'event', 'brief'],

    'document':        ['article', 'op-ed', 'column', 'editorial', 'blog post', 'blog', 'news story', 'opinion piece', 'oped', 'journal','news','chronicle',
                        'story','book','letter','report','flier','flyer','document','magazine','newspaper','news paper','study','text','chart','autobiography',
                        'statement','quote','testimony','press release', 'news release', 'press statement', 'press conference', 'news conference', 'press',
                        'email', 'blast', 'mail', 'mailer', 'chain email', 'newsletter', 'fundraising email', 'inbox'],

    'media':           ['a ', 'an ', 'on ', 'the ','abc', 'cnn', 'politico','tv interview', 'radio', 'broadcast', 'show', 'segment', 'program', 
                        'tv', 'documentary','live','monologue','comedy','video', 'web video', 'campaign video','image', 'photo', 'screenshot', 
                        'viral image','episode', 'phone'],

    'location':        ['iowa','ohio','florida','fla','miami','new ','ville', 'las vegas', 'tampa', 'cleveland', 'denver', 'manchester', 'orlando',
                        'washington','nashville','austin','oxford','houston','dallas','detroit','carolina', 'ny', 'nh', 'nc', 'city', 'st', 'beach',
                        'ill', 'valley','calif','india','nm','texas','san','mo','pa','va','us']
}

VERACITY_COUNT_COLUMNS = [
    'true_counts', 'mostly_true_counts', 'half_true_counts',
    'mostly_false_counts', 'false_counts', 'pants_on_fire_counts'
]

def map_context_generic(ctx: str) -> str:
    """
    Mapeado de contexto a su categoría concreta:
        A patir de una serie de categorías base
        se mapean los posibles valores de la
        columna.
    """
    text = ctx.lower()
    for cat, keywords in GENERIC_CTX.items():
        for kw in keywords:
            if kw in text:
                return cat
    return 'location'

def prepare_context_generic(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesar la variable context:
        Limpieza texto context
        Mapeo a categorias
        Creación de variables dummies
        Concatenación con el dataframe 
    """
    df = df_original.copy()
    df['context'] = df['context'].apply(clean_text_basic)
    # Mapear a categorias
    df['context_cat'] = df['context'].apply(map_context_generic)
    # One-hot
    dummies = pd.get_dummies(df['context_cat'], prefix='ctx', drop_first=True)

    df_result = pd.concat([df, dummies], axis=1)
    return df_result


def add_count_proportions(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Refactor de las variables que cuentan el número de cada tipo de noticia
    a variables que representan la proporción de cada una de ellas:
        Creación de una variable que agrupa el número total de noticias
        de cada speaker
        Se indica como nan aquel que no tiene noticias
        Se generan las proporciones en función de las noticias totales
        Rellenamos los valores NaN con 0
        Borramos las variables count
    """
    df = df_original.copy()

    df['total_counts'] = df[VERACITY_COUNT_COLUMNS].sum(axis=1)

    # Si algún speaker tuviera total_counts == 0, le ponemos null para que no falle
    df['total_counts_adj'] = df['total_counts'].replace(0, np.nan)

    # Creamos las nuevas columnas de proporciones
    for col in VERACITY_COUNT_COLUMNS:
        prop_col = col.replace('_counts', '_prop')
        df[prop_col] = df[col] / df['total_counts_adj']

    # Rellenar NaN con 0
    prop_cols = [c.replace('_counts', '_prop') for c in VERACITY_COUNT_COLUMNS]
    df[prop_cols] = df[prop_cols].fillna(0)

    df.drop(columns=VERACITY_COUNT_COLUMNS, inplace=True)
    return df


def clean_topic(tok):
    """
    Limpia cada dato:
        Union de espacios
        Conversión de - a _
        Conversión de ' ' a _
        Quitar caracteres raros a excepción de _
    """
    tok = tok.strip()
    tok = tok.replace('-', '_')       # Conversión: fact-check → fact_check
    tok = tok.replace(' ', '_')       # Conversión: after the fact → after_the_fact
    tok = re.sub(r'[^a-z0-9_ñáéíóúü]', '', tok)
    return tok

def process_subject(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Procesado de temas en el dataset:
        Conversión a minúscula y división en lista en función de ;
        Aplicar limpieza de los datos a la lista de temas
        Reconstruye un texto donde cada tema es un token
    """
    df = df_original.copy()

    df['subject_list'] = (
        df['subject']
        .str.lower()
        .str.split(';')
    )

    df['subject_list'] = df['subject_list'].apply(lambda lst: [clean_topic(t) for t in lst])
    df['subject'] = df['subject_list'].apply(lambda lst: ' '.join(lst))

    return df

def remove_unused_columns(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Borrado de las columnas que no son necesarias para el dataset
    """
    df = df_original.copy()

    df.drop(columns=['speaker','id','context','context_cat','subject_list','total_counts_adj','date'], inplace=True)

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

# 0   id                    22962 non-null  int64   -> Remove when finish   [BORRAR]   
# 1   label                 22962 non-null  int64   -> Target      
# 2   statement             22962 non-null  object  -> As classic text      
# 3   date                  22962 non-null  datetime64[ns] -> Extracted year, month, day and weekday as numeric values [BORRAR]
# 4   subject               22962 non-null  object  -> Convertido el texto      
# 5   speaker               22962 non-null  object  -> No aporta información, además ya tenemos algunas columnas derivadas de esta [BORRAR]    
# 6   speaker_description   22962 non-null  object  -> As classic text      
# 7   true_counts           22962 non-null  int64   -> Creadas variables proporcion correspondiente [BORRAR]    
# 8   mostly_true_counts    22962 non-null  int64   -> Creadas variables proporcion correspondiente [BORRAR]    
# 9   half_true_counts      22962 non-null  int64   -> Creadas variables proporcion correspondiente [BORRAR]    
# 10  mostly_false_counts   22962 non-null  int64   -> Creadas variables proporcion correspondiente [BORRAR]    
# 11  false_counts          22962 non-null  int64   -> Creadas variables proporcion correspondiente [BORRAR]     
# 12  pants_on_fire_counts  22962 non-null  int64   -> Creadas variables proporcion correspondiente [BORRAR]    
# 13  context               22962 non-null  object  -> Categoric with dummies  [BORRAR]  junto con 'context_cat'   
# 14  justification         22962 non-null  object  -> As classic text

if __name__ == '__main__':
    df = load_data('liar2_no_null_data.csv')

    df = prepare_date_for_model(df)

    df = prepare_text_classic(df, ['statement','speaker_description','justification'])

    df = prepare_context_generic(df)
    
    df = add_count_proportions(df)

    df = process_subject(df)

    df = remove_unused_columns(df)

    df.to_csv('liar2_preprocessed.csv',index=False)
