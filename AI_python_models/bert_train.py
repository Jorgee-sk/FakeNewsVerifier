import os

os.environ["HF_HUB_CACHE"] = r"E:\hug_cache\hub"
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
local_dir = r"E:\hug_cache\hub\models--bert-base-uncased--manual"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.amp import autocast, GradScaler
from bert_utils import load_and_split_data, Liar2Dataset
import shap

# ----------------------------
# CONFIGURACIÓN DE PARÁMETROS
# ----------------------------

TEXT_CSV = 'liar2_for_bert_text.csv'
META_CSV = 'liar2_for_bert_meta.csv'
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 352
LOG_INTERVAL = 50
BATCH_SIZE = 6
ACCUM_STEPS = 4
NUM_EPOCHS = 8
LEARNING_RATE = 1e-5
WARMUP_PROPORTION = 0.1
GAMMA = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 2
TEMPERATURE = 1.0
CLAMP_VAL = 10.0

# ----------------------------
# DEFINICIÓN DEL MODELO
# ----------------------------
class BertWithMetadata(nn.Module):
    def __init__(self, pretrained_model_name: str, metadata_dim: int,
                 hidden_metadata: int = 256, num_labels: int = 6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(local_dir, local_files_only=True)
        bert_hidden_size = self.bert.config.hidden_size

        # Metadata projection
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, hidden_metadata),
            nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(hidden_metadata, hidden_metadata),
            nn.ReLU(), nn.Dropout(0.05)
        )
        for module in self.metadata_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

        self.dropout = nn.Dropout(0.05)
        self.classifier = nn.Linear(bert_hidden_size + hidden_metadata, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, metadata):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        meta_repr = self.metadata_proj(metadata)
        combined = torch.cat([pooled_output, meta_repr], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

# ---------------------------
#        FOCAL LOSS
# ---------------------------
def focal_loss(logits, labels, gamma=GAMMA, alpha=None, reduction='mean'):
    """
    logits: [batch_size, num_classes]
    labels: [batch_size] con valores 0..num_classes-1
    alpha: float o tensor [num_classes] para ponderar clases (puede ser None)
    gamma: factor de focalización
    """
    ce = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce)                                        

    if alpha is None:
        alpha_t = 1.0
    elif isinstance(alpha, (float, int)):
        alpha_t = alpha
    else:
        # si alpha es tensor de pesos por clase
        alpha_t = alpha[labels]

    loss = alpha_t * (1 - pt)**gamma * ce

    return loss.mean() if reduction=='mean' else loss.sum()
    
def criterion(logits, labels):
    return focal_loss(logits, labels,
                      gamma=GAMMA,
                      alpha=alpha_tensor,
                      reduction='mean')


# ----------------------------
# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ----------------------------

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    running_loss, running_correct, running_count = 0.0, 0, 0
    num_batches = len(dataloader)
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward en mixed-precision
        with autocast(device_type="cuda"):
            logits = model(input_ids, attention_mask, metadata)

        # Validación de alineación: imprime labels y predicciones
        if step == 0:
            preds_check = torch.argmax(logits.float(), dim=1)
            print(f"Batch 0 labels: {labels.tolist()}")
            print(f"Batch 0 preds : {preds_check.tolist()}")

        # Escala y recorta logits
        logits_fp32 = logits.float() / TEMPERATURE
        logits_fp32 = torch.clamp(logits_fp32, -CLAMP_VAL, CLAMP_VAL)

        # Cálculo de focal loss en FP32
        ce_sample = F.cross_entropy(logits_fp32, labels, reduction='none')
        pt = torch.exp(-ce_sample)
        alpha_t = alpha_tensor[labels]
        focal_factor = (1 - pt)**GAMMA
        fl_sample = alpha_t * focal_factor * ce_sample
        raw_loss = fl_sample.mean()

        if step == 0:
            print(f"[Debug Ep{epoch+1}] CE: {ce_sample.mean():.4f}, pt: {pt.mean():.4f}")

        loss = raw_loss / ACCUM_STEPS
        # Backward escalado
        scaler.scale(loss).backward()

        # Accumulation step
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += raw_loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Acumular métricas “running” para log
        running_loss += raw_loss.item()
        running_correct += (preds == labels.cpu().numpy()).sum()
        running_count += len(labels)

        if ((step + 1) % LOG_INTERVAL == 0) or ((step + 1) == num_batches):
            avg_run_loss = running_loss / LOG_INTERVAL
            avg_run_acc = running_correct / running_count
            print(f"  [Ep {epoch+1}/{NUM_EPOCHS}] "
                  f"Batch {step+1}/{num_batches} — "
                  f"Avg Loss (últ. {LOG_INTERVAL}): {avg_run_loss:.4f} — "
                  f"Avg Acc (últ. {LOG_INTERVAL}): {avg_run_acc:.4f}")
            running_loss = 0.0
            running_correct = 0
            running_count = 0

    avg_loss = total_loss / num_batches
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            logits = model(input_ids, attention_mask, metadata)
            # FP32 logits with temperature and clamp
            logits_fp32 = logits.float() / TEMPERATURE
            logits_fp32 = torch.clamp(logits_fp32, -CLAMP_VAL, CLAMP_VAL)
            # Focal loss per sample
            ce_sample = F.cross_entropy(logits_fp32, labels, reduction='none')
            pt = torch.exp(-ce_sample)
            alpha_t = alpha_tensor[labels]
            focal_factor = (1 - pt)**GAMMA
            fl_sample = alpha_t * focal_factor * ce_sample
            raw_loss = fl_sample.mean()

            total_loss += raw_loss.item()
            preds = torch.argmax(logits_fp32, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


# ----------------------------
# SHAP: EXPLICACIÓN MIXTA (TEXTO + METADATA)
# ----------------------------

def shap_explain(model, tokenizer, df_text, df_meta, device, background_size=50):
    """
    Prepara un explainer SHAP que explique simultáneamente tokens y features tabulares.
    - model: modelo entrenado (BertWithMetadata)
    - tokenizer: AutoTokenizer
    - df_text: DataFrame de validación con columnas ['text_for_bert', 'label']
    - df_meta: DataFrame de validación con features numéricas/categóricas
    - background_size: cuántos ejemplos tomar para el fondo
    Devuelve: explainer listo para calcular shap_values(inputs) donde inputs = [(texto, array_meta), ...]
    """
    # Selección aleatoria de background
    np.random.seed(200900)

    df_text = df_text.reset_index(drop=True)
    df_meta = df_meta.reset_index(drop=True)

    idxs_bg = np.random.choice(len(df_text), size=background_size, replace=False)
    bg_texts = [df_text.iloc[i]['text_for_bert'] for i in idxs_bg]
    bg_metas  = np.stack([df_meta.iloc[i].values.astype('float32') for i in idxs_bg],
                         axis=0)
    
    masker_text = shap.maskers.Text(tokenizer)
    masker_tab = shap.maskers.Independent(data=bg_metas)
    
    def f_mixed(texts, metas_np):
        enc = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        metadata_tensor= torch.tensor(metas_np, dtype=torch.float).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask, metadata_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    # Crear explainer
    explainer = shap.Explainer(
        f_mixed,
        [masker_text, masker_tab],
        [bg_texts, bg_metas]    # <-- aquí cada sub-masker toma su propio fondo
    )
    return explainer


# ----------------------------
# BLOQUE PRINCIPAL
# ----------------------------

if __name__ == '__main__':
    # Cargar y dividir datos
    df_text_train, df_meta_train, df_text_val, df_meta_val, df_text_test, df_meta_test = \
        load_and_split_data(TEXT_CSV, META_CSV,
                            test_size=0.20, val_size=0.50, random_state=200900)

    y_train = df_text_train['label'].values  

    # Cálculo de pesos y sampler balanceado
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float, device=DEVICE)
    alpha_tensor /= alpha_tensor.mean()
    alpha_tensor[1] *= 1.7
    alpha_tensor[2] *= 0.8
    alpha_tensor[3] *= 1.2
    alpha_tensor /= alpha_tensor.mean()
    sample_weights = alpha_tensor[y_train].cpu().numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Inicializar tokenizer y datasets
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

    train_dataset = Liar2Dataset(df_text_train, df_meta_train, tokenizer, MAX_LEN)
    val_dataset = Liar2Dataset(df_text_val, df_meta_val, tokenizer, MAX_LEN)
    test_dataset = Liar2Dataset(df_text_test, df_meta_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inicializar modelo
    metadata_dim = df_meta_train.shape[1]
    model = BertWithMetadata(pretrained_model_name=MODEL_NAME,
                             metadata_dim=metadata_dim,
                             hidden_metadata=256,
                             num_labels=6).to(DEVICE)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    total_steps = len(train_loader) * NUM_EPOCHS // ACCUM_STEPS

    num_warmup_steps = int(WARMUP_PROPORTION * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
 
    # Instanciar GradScaler para FP16
    scaler = GradScaler()

    # Bucle de entrenamiento / validación
    best_val_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, epoch)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_state.bin')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"No hubo mejora en validación en las últimas {PATIENCE} épocas; deteniendo.")
            break

    # Cargar mejor modelo para evaluación final
    model.load_state_dict(torch.load('best_model_state.bin'))
    model.to(DEVICE)

    # Evaluación en test
    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, DEVICE)
    print(f"\n=== TEST SET ===")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}\n")

    # Métricas detalladas
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            metadata = batch['metadata'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           metadata=metadata)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4,5], normalize='true')
    
    label_names = ['pants-fire','false','barely-true','half-true','mostly-true','true']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de confusión')
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


    print("Reporte por clase:")
    print(classification_report(all_labels, all_preds, digits=4,
                                target_names=['pants-fire','false',
                                              'barely-true','half-true',
                                              'mostly-true','true']))

    # Guardar modelo y tokenizer finales
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    model.bert.save_pretrained('saved_model')
    tokenizer.save_pretrained('saved_model')

    # Ejemplo de explicaciones con SHAP (sobre val set)
    print("\nGenerando explicaciones SHAP en 3 ejemplos de validación...")
    explainer = shap_explain(model, tokenizer, df_text_val, df_meta_val, DEVICE)

    examples_to_explain = 3

    texts = [df_text_val.iloc[i]['text_for_bert'] for i in range(examples_to_explain)]
    metas  = np.vstack([df_meta_val.iloc[i].values.astype('float32') for i in range(examples_to_explain)])
    
    print(texts)
    print(metas.shape)
    # Llama al explainer con DOS argumentos
    shap_values = explainer(texts, metas)

    shap.plots.text(shap_values[0])
    shap.plots.bar(shap_values[0])

    # Opcional: guardar shap_values para análisis futuro
    np.save('shap_values.npy', shap_values.values)

    print("=== FIN DEL PROCESO ===")