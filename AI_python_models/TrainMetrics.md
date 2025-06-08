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

============================================

=========== TRANSFORMERS (BERT) ============

============================================

**Primera iteración del modelo** 
Epoch 6/6 | Train Loss: 0.3331 | Train Acc: 0.4746 | Train F1: 0.4608 || Val Loss: 1.4534 | Val Acc: 0.3911 | Val F1: 0.3930

=== TEST SET ===
Test Loss: 1.4860 | Test Acc: 0.3875 | Test F1: 0.3917

Matriz de Confusión (filas=verdadero, columnas=predicho):
[[151  55  24  29  10  34]
 [232  93  93  86  50 107]
 [ 22  21 172  67  39  40]
 [  5  14  34 191  68  59]
 [  4   7  20  94 139  79]
 [ 11   9   5  52  37 144]]

Reporte por clase:
              precision    recall  f1-score   support

  pants-fire     0.3553    0.4983    0.4148       303
       false     0.4673    0.1407    0.2163       661
 barely-true     0.4943    0.4765    0.4852       361
   half-true     0.3680    0.5148    0.4292       371
 mostly-true     0.4052    0.4052    0.4052       343
        true     0.3110    0.5581    0.3994       258

    accuracy                         0.3875      2297
   macro avg     0.4002    0.4323    0.3917      2297
weighted avg     0.4139    0.3875    0.3679      2297

-> Aumentado MAX_LEN
-> Añadida nueva capa a metadatos y reducido el dropout a 0
-> Modificado el tamaño de batch de 4 a 6
-> Añadido focal loss y oversampling y configurados ciertos parámetros

**Segunda iteración del modelo** 
Epoch 1/6 | Train Loss: 2.7200 | Train Acc: 0.2401 | Train F1: 0.2230 || Val Loss: 1.3989 | Val Acc: 0.3001 | Val F1: 0.2908
Epoch 2/6 | Train Loss: 0.2914 | Train Acc: 0.4442 | Train F1: 0.4264 || Val Loss: 1.1375 | Val Acc: 0.4020 | Val F1: 0.4032
Epoch 3/6 | Train Loss: 0.2128 | Train Acc: 0.5539 | Train F1: 0.5350 || Val Loss: 0.9445 | Val Acc: 0.4983 | Val F1: 0.4999
Epoch 4/6 | Train Loss: 0.1706 | Train Acc: 0.6060 | Train F1: 0.5895 || Val Loss: 0.7915 | Val Acc: 0.5122 | Val F1: 0.5216
Epoch 5/6 | Train Loss: 0.1447 | Train Acc: 0.6435 | Train F1: 0.6299 || Val Loss: 0.7590 | Val Acc: 0.5196 | Val F1: 0.5290
Epoch 6/6 | Train Loss: 0.1278 | Train Acc: 0.6701 | Train F1: 0.6552 || Val Loss: 0.7203 | Val Acc: 0.5422 | Val F1: 0.5508 

=== TEST SET ===
Test Loss: 0.6905 | Test Acc: 0.5620 | Test F1: 0.5686

Matriz de Confusión (filas=verdadero, columnas=predicho):
[[224  41  21   4   5   8]
 [188 217 129  34  25  68]
 [ 16  23 238  37  31  16]
 [  5   3  62 201  78  22]
 [  1   2  19  42 208  71]
 [  5   6   6   9  29 203]]

Reporte por clase:
              precision    recall  f1-score   support

  pants-fire     0.5103    0.7393    0.6038       303
       false     0.7432    0.3283    0.4554       661
 barely-true     0.5011    0.6593    0.5694       361
   half-true     0.6147    0.5418    0.5759       371
 mostly-true     0.5532    0.6064    0.5786       343
        true     0.5232    0.7868    0.6285       258

    accuracy                         0.5620      2297
   macro avg     0.5743    0.6103    0.5686      2297
weighted avg     0.6006    0.5620    0.5502      2297

-> Aumentado warmup a 0.15
-> Subido dropout de 0 a 0.1
-> Balanceado peso en alpha de la clase false para intentar mejorar el recall

**Tercera iteracion**
Epoch 6/6 | Train Loss: 1.1049 | Train Acc: 0.4611 | Train F1: 0.4389 || Val Loss: 2.5282 | Val Acc: 0.4669 | Val F1: 0.4572

=== TEST SET ===
Test Loss: 2.4115 | Test Acc: 0.4750 | Test F1: 0.4655

-> Malisimo , probamos otras cosas
-> Aumentamos epochs a 8
-> Aumentamos LR a 3e-5
-> Bajamos dropout de 0.1 a 0.05 en metadata y a 0 en clasif
-> Balanceado peso en alpha de la clase false para intentar mejorar el recall subido a .8

**Cuarta iteracion**
=== TEST SET ===
Test Loss: 1.6658 | Test Acc: 0.4902 | Test F1: 0.4931

Reporte por clase:
              precision    recall  f1-score   support

  pants-fire     0.4319    0.7855    0.5574       303
       false     0.7143    0.1210    0.2070       661
 barely-true     0.4147    0.6870    0.5172       361
   half-true     0.5671    0.5013    0.5322       371
 mostly-true     0.5325    0.5977    0.5632       343
        true     0.5232    0.6550    0.5818       258

    accuracy                         0.4902      2297
   macro avg     0.5306    0.5579    0.4931      2297
weighted avg     0.5576    0.4902    0.4498      2297


-> Mejora general pero false baja terrible , probamos otras cosas
-> Aumentamos epochs a 8
-> Reducimos LR a 1e-5
-> Balanceado peso en alpha de la clase false para intentar mejorar el recall subido a 1.5
-> Reducido Gamma a 1

**Quinta iteracion**
=== TEST SET ===
Test Loss: 2.3318 | Test Acc: 0.5686 | Test F1: 0.5714

Reporte por clase:
              precision    recall  f1-score   support

  pants-fire     0.5648    0.6040    0.5837       303
       false     0.6521    0.4962    0.5636       661
 barely-true     0.4566    0.6704    0.5432       361
   half-true     0.6654    0.4663    0.5483       371
 mostly-true     0.5753    0.5569    0.5659       343
        true     0.5431    0.7326    0.6238       258

    accuracy                         0.5686      2297
   macro avg     0.5762    0.5877    0.5714      2297
weighted avg     0.5883    0.5686    0.5677      2297

**Sexta iteracion**
Epoch 8/8 | Train Loss: 0.4359 | Train Acc: 0.6644 | Train F1: 0.6647 || Val Loss: 0.5713 | Val Acc: 0.5976 | Val F1: 0.5997

=== TEST SET ===
Test Loss: 0.5367 | Test Acc: 0.6152 | Test F1: 0.6161

Reporte por clase:
              precision    recall  f1-score   support

  pants-fire     0.5647    0.6337    0.5972       303
       false     0.6495    0.5915    0.6192       661
 barely-true     0.7208    0.5291    0.6102       361
   half-true     0.5315    0.7278    0.6143       371
 mostly-true     0.6233    0.5452    0.5816       343
        true     0.6454    0.7054    0.6741       258

    accuracy                         0.6152      2297
   macro avg     0.6225    0.6221    0.6161      2297
weighted avg     0.6261    0.6152    0.6146      2297