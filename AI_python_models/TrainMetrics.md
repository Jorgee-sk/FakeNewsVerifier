==========================================
      PRIMER ENTRENAMIENTO DE MODELOS
==========================================


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