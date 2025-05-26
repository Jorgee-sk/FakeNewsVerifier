import pandas as pd
from sklearn.model_selection import train_test_split


def data_split():
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

    return X_train, X_val, X_test, y_train, y_val, y_test