import pandas as pd
import re
import numpy as np

def clean_text_basic_for_bert(text: str) -> str:
    # Si se va a usar BERT cased, quitar el .lower()
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def prepare_text_bert(df_original: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """
    Pipeline clásico para textos:
        Limpieza básica
        Cálculo de longitudes
    """
    df = df_original.copy()

    for column_name in column_names:
        cleanColumn = 'clean_'+column_name
        lenCharsColumn = column_name+'_len_chars'
        lenWordsColumn = column_name+'_len_words'

        df[cleanColumn] = df[column_name].apply(clean_text_basic_for_bert) 
        # Extrae longitud de texto como feature numérica
        df[lenCharsColumn] = df[cleanColumn].str.len()
        df[lenWordsColumn] = df[cleanColumn].str.split().str.len()

        df.drop(columns=[column_name], inplace=True)

    return df


# Context: mapea a categorías y saca dummies

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
    df['context'] = df['context'].apply(clean_text_basic_for_bert)
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

def combine_texts(row):
    subj = row['subject']
    spk =  row['clean_speaker_description']
    stmt = row['clean_statement']
    just = row['clean_justification']
    return f"{stmt} [SEP] {subj} [SEP] {spk} [SEP] {just} "

# ------------------------------------------------------------------
#
# ------------------------------------------------------------------
df = pd.read_csv('liar2_no_null_data.csv')

df = prepare_text_bert(df, ['statement','speaker_description','justification'])

# Metadatos para BERT
# Fecha
df['date'] = pd.to_datetime(df['date'], format='%B %d, %Y', errors='coerce')
df['year'] = df['date'].dt.year.astype(int)
df['month'] = df['date'].dt.month.astype(int)
df['day'] = df['date'].dt.day.astype(int)
df['dayofweek'] = df['date'].dt.dayofweek.astype(int)
    
# Subject como texto
df['subject_list'] = df['subject'].str.lower().str.split(';').apply(lambda lst: [t.strip().replace('-', '_').replace(' ', '_') for t in lst])
df['subject'] = df['subject_list'].apply(lambda lst: ' '.join(lst))

# Context
df = prepare_context_generic(df)

# Proporciones (true_prop, mostly_true_prop, etc.) con add_count_proportions
df = add_count_proportions(df)

# BERT + metadata:
# un CSV para el texto:
df.drop(columns=['speaker','id','context','context_cat','subject_list','total_counts_adj','date'], inplace=True)

df['text_for_bert'] = df.apply(combine_texts, axis=1)
df[['text_for_bert', 'label']].to_csv('liar2_for_bert_text.csv', index=False)

# otro CSV con todas las columnas numéricas/categóricas:
numeric_cols = [
    'year', 'month','day','dayofweek',
    'true_prop', 'mostly_true_prop', 'half_true_prop',
    'mostly_false_prop', 'false_prop', 'pants_on_fire_prop','total_counts',
    'statement_len_words', 'statement_len_chars',
    'speaker_description_len_words', 'speaker_description_len_chars',
    'justification_len_words', 'justification_len_chars'
]
df_meta = df[numeric_cols].astype('float32')
df_meta.to_csv('liar2_for_bert_meta.csv', index=False)