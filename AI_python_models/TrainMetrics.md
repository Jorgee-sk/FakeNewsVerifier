===========================================================================================================
      ENTRENAMIENTO DE MODELOS BASE (PIPELINES BASE SIN OVERSAMPLING - SOLO TEST DE QUE FUNCIONAN)
===========================================================================================================


Mejores params: {'clf__C': 0.1}

=== SVM best params ===
 {'clf__C': 0.1}

=== SVM Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.63      0.59      0.61       455
           1       0.68      0.66      0.67       991
           2       0.65      0.59      0.62       540
           3       0.62      0.57      0.59       556
           4       0.57      0.61      0.59       514
           5       0.49      0.63      0.55       388

    accuracy                           0.61      3444
   macro avg       0.61      0.61      0.60      3444
weighted avg       0.62      0.61      0.62      3444

Mejores params: {'clf__C': 1}

=== LogReg best params ===
 {'clf__C': 1}

=== LogReg Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.59      0.64      0.61       455
           1       0.69      0.61      0.65       991
           2       0.62      0.59      0.61       540
           3       0.61      0.58      0.59       556
           4       0.57      0.60      0.58       514
           5       0.49      0.61      0.54       388

    accuracy                           0.61      3444
   macro avg       0.59      0.61      0.60      3444
weighted avg       0.61      0.61      0.61      3444

Mejores params: {'clf__max_depth': 20, 'clf__n_estimators': 200}

=== RF best params ===
 {'clf__max_depth': 20, 'clf__n_estimators': 200}

=== RF Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.71      0.43      0.53       455
           1       0.62      0.73      0.67       991
           2       0.69      0.55      0.61       540
           3       0.56      0.56      0.56       556
           4       0.54      0.60      0.57       514
           5       0.47      0.56      0.51       388

    accuracy                           0.59      3444
   macro avg       0.60      0.57      0.58      3444
weighted avg       0.61      0.59      0.59      3444

Mejores params: {'clf__learning_rate': 0.1, 'clf__n_estimators': 100}

=== GB best params ===
 {'clf__learning_rate': 0.1, 'clf__n_estimators': 100}

=== GB Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.68      0.47      0.55       455
           1       0.60      0.77      0.68       991
           2       0.72      0.54      0.62       540
           3       0.61      0.56      0.58       556
           4       0.58      0.59      0.58       514
           5       0.52      0.54      0.53       388

    accuracy                           0.61      3444
   macro avg       0.62      0.58      0.59      3444
weighted avg       0.62      0.61      0.61      3444

Mejores params: {'clf__alpha': 1.0}

=== NB best params ===
 {'clf__alpha': 1.0}

=== NB Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.46      0.56      0.51       455
           1       0.64      0.34      0.45       991
           2       0.45      0.54      0.49       540
           3       0.45      0.46      0.45       556
           4       0.43      0.48      0.46       514
           5       0.33      0.48      0.39       388

    accuracy                           0.46      3444
   macro avg       0.46      0.48      0.46      3444
weighted avg       0.49      0.46      0.46      3444


**Mejor modelo: SVM**

=== Test final ===
              precision    recall  f1-score   support

           0       0.60      0.57      0.58       454
           1       0.65      0.64      0.65       991
           2       0.64      0.57      0.60       541
           3       0.62      0.58      0.60       557
           4       0.59      0.60      0.59       515
           5       0.48      0.65      0.55       387

    accuracy                           0.60      3445
   macro avg       0.60      0.60      0.60      3445
weighted avg       0.61      0.60      0.60      3445

====================================================================================
                     PRIMER ENTRENAMIENTO DE MODELOS 
            (+Transformadores de texto que incluyen characters)
   (+Extra stopwords para borrar palabras que no aportan mucho significado)
      (+Uso de SMOTE para realizar oversampling de los datos del modelo)
====================================================================================

Mejores params: {'clf__C': 0.01}

=== SVM best params ===
 {'clf__C': 0.01}

=== SVM Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.53      0.63      0.58       455
           1       0.71      0.56      0.63       991
           2       0.63      0.61      0.62       540
           3       0.64      0.59      0.61       556
           4       0.62      0.61      0.61       514
           5       0.46      0.69      0.55       388

    accuracy                           0.60      3444
   macro avg       0.60      0.62      0.60      3444
weighted avg       0.62      0.60      0.61      3444

Mejores params: {'clf__C': 1}

=== LogReg best params ===
 {'clf__C': 1}

=== LogReg Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.56      0.58      0.57       455
           1       0.66      0.63      0.64       991
           2       0.64      0.61      0.62       540
           3       0.63      0.59      0.61       556
           4       0.61      0.61      0.61       514
           5       0.51      0.62      0.56       388

    accuracy                           0.61      3444
   macro avg       0.60      0.61      0.60      3444
weighted avg       0.61      0.61      0.61      3444

Mejores params: {'clf__max_depth': 10, 'clf__n_estimators': 200}

=== RF best params ===
 {'clf__max_depth': 10, 'clf__n_estimators': 200}

=== RF Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.56      0.59      0.57       455
           1       0.72      0.53      0.61       991
           2       0.62      0.55      0.59       540
           3       0.47      0.47      0.47       556
           4       0.50      0.57      0.53       514
           5       0.41      0.65      0.51       388

    accuracy                           0.55      3444
   macro avg       0.55      0.56      0.55      3444
weighted avg       0.58      0.55      0.55      3444

Mejores params: {'clf__learning_rate': 0.1, 'clf__n_estimators': 100}

=== GB best params ===
 {'clf__learning_rate': 0.1, 'clf__n_estimators': 100}

=== GB Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.55      0.55      0.55       455
           1       0.68      0.60      0.64       991
           2       0.66      0.55      0.60       540
           3       0.50      0.52      0.51       556
           4       0.56      0.57      0.57       514
           5       0.44      0.63      0.52       388

    accuracy                           0.57      3444
   macro avg       0.57      0.57      0.56      3444
weighted avg       0.59      0.57      0.58      3444

Mejores params: {'clf__alpha': 0.5}

=== NB best params ===
 {'clf__alpha': 0.5}

=== NB Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.43      0.57      0.49       455
           1       0.60      0.31      0.41       991
           2       0.43      0.49      0.46       540
           3       0.40      0.44      0.42       556
           4       0.39      0.44      0.41       514
           5       0.35      0.45      0.39       388

    accuracy                           0.43      3444
   macro avg       0.43      0.45      0.43      3444
weighted avg       0.46      0.43      0.43      3444


**Mejor modelo: LogReg**

=== Test final ===
              precision    recall  f1-score   support

           0       0.59      0.58      0.58       454
           1       0.62      0.62      0.62       991
           2       0.58      0.56      0.57       541
           3       0.60      0.59      0.60       557
           4       0.58      0.57      0.58       515
           5       0.54      0.63      0.58       387

    accuracy                           0.59      3445
   macro avg       0.59      0.59      0.59      3445
weighted avg       0.59      0.59      0.59      3445

====================================================================================
            SEGUNDO ENTRENAMIENTO DE MODELOS CON TECNICAS TRADICIONALES
      Voy a probar a reducir el min_df en los distintos transformadores de texto
      Además voy a añadir el stage de chi2 para eliminar palabras menos relevantes
====================================================================================
Mejores params: {'clf__C': 0.01}

=== SVM best params ===
 {'clf__C': 0.01}

=== SVM Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.54      0.67      0.60       303
           1       0.69      0.56      0.62       660
           2       0.66      0.64      0.65       360
           3       0.66      0.60      0.63       371
           4       0.66      0.64      0.65       343
           5       0.49      0.70      0.58       259

    accuracy                           0.62      2296
   macro avg       0.62      0.63      0.62      2296
weighted avg       0.63      0.62      0.62      2296

Mejores params: {'clf__C': 0.1}

=== LogReg best params ===
 {'clf__C': 0.1}

=== LogReg Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.54      0.66      0.60       303
           1       0.69      0.56      0.62       660
           2       0.63      0.64      0.63       360
           3       0.66      0.60      0.63       371
           4       0.66      0.65      0.65       343
           5       0.51      0.68      0.58       259

    accuracy                           0.62      2296
   macro avg       0.62      0.63      0.62      2296
weighted avg       0.63      0.62      0.62      2296

Mejores params: {'clf__max_depth': None, 'clf__n_estimators': 200}

=== RF best params ===
 {'clf__max_depth': None, 'clf__n_estimators': 200}

=== RF Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.71      0.48      0.57       303
           1       0.61      0.75      0.68       660
           2       0.71      0.54      0.61       360
           3       0.66      0.56      0.61       371
           4       0.65      0.59      0.62       343
           5       0.49      0.71      0.58       259

    accuracy                           0.62      2296
   macro avg       0.64      0.61      0.61      2296
weighted avg       0.64      0.62      0.62      2296

Mejores params: {'clf__alpha': 0.5}

=== NB best params ===
 {'clf__alpha': 0.5}

=== NB Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.48      0.64      0.55       303
           1       0.58      0.32      0.41       660
           2       0.44      0.52      0.48       360
           3       0.40      0.42      0.41       371
           4       0.44      0.48      0.46       343
           5       0.36      0.47      0.41       259

    accuracy                           0.45      2296
   macro avg       0.45      0.47      0.45      2296
weighted avg       0.47      0.45      0.45      2296

Mejores params: {'clf__estimator__C': 0.1}

=== LogRegCal best params ===
 {'clf__estimator__C': 0.1}

=== LogRegCal Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.53      0.69      0.60       303
           1       0.72      0.53      0.61       660
           2       0.65      0.64      0.64       360
           3       0.66      0.60      0.63       371
           4       0.67      0.64      0.65       343
           5       0.48      0.74      0.59       259

    accuracy                           0.62      2296
   macro avg       0.62      0.64      0.62      2296
weighted avg       0.64      0.62      0.62      2296

Mejores params: {'clf__learning_rate': 0.01, 'clf__max_depth': None, 'clf__max_iter': 400}

=== HGB best params ===
 {'clf__learning_rate': 0.01, 'clf__max_depth': None, 'clf__max_iter': 400}

=== HGB Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.70      0.58      0.63       303
           1       0.64      0.77      0.69       660
           2       0.74      0.62      0.67       360
           3       0.71      0.64      0.67       371
           4       0.68      0.62      0.65       343
           5       0.58      0.68      0.63       259

    accuracy                           0.67      2296
   macro avg       0.67      0.65      0.66      2296
weighted avg       0.67      0.67      0.67      2296

Mejores params: {'clf__learning_rate': 0.1, 'clf__max_depth': 6, 'clf__n_estimators': 200}

=== XGBoost best params ===
 {'clf__learning_rate': 0.1, 'clf__max_depth': 6, 'clf__n_estimators': 200}

=== XGBoost Classification Report on VAL ===
              precision    recall  f1-score   support

           0       0.70      0.52      0.60       303
           1       0.61      0.79      0.69       660
           2       0.74      0.59      0.66       360
           3       0.72      0.63      0.67       371
           4       0.67      0.62      0.65       343
           5       0.59      0.66      0.62       259

    accuracy                           0.66      2296
   macro avg       0.67      0.64      0.65      2296
weighted avg       0.67      0.66      0.66      2296


**Mejor modelo: XGBoost**

=== Test final ===
              precision    recall  f1-score   support

           0       0.73      0.51      0.60       303
           1       0.61      0.80      0.69       661
           2       0.72      0.61      0.66       361
           3       0.72      0.61      0.66       371
           4       0.67      0.59      0.63       343
           5       0.56      0.65      0.60       258

    accuracy                           0.65      2297
   macro avg       0.67      0.63      0.64      2297
weighted avg       0.67      0.65      0.65      2297