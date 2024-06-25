import os
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# ====================== 설정 =========================

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = 'monologg/kobert'
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 2

# ===================== 유틸 함수 ====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_to_kobert_inputs(text_list, max_len, tokenizer):
    input_ids, attention_masks, token_type_ids = [], [], []
    for text in text_list:
        text = "" if pd.isnull(text) else text
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(token_type_ids, dtype=torch.long)
    )

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def label_smoothing_loss(logits, labels, smoothing=0.0):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    return (confidence * nll_loss + smoothing * smooth_loss).mean()

# ===================== 학습 메인 =====================

def main():
    set_seed(42)

    df = pd.read_csv('../data/processed_output.csv', encoding='utf-8-sig')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(
        df['combined_text'].values, MAX_LEN, tokenizer
    )
    labels = torch.tensor(df['blog_is_promotional'].values, dtype=torch.long)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
    )

    train_data = TensorDataset(
        input_ids[train_idx], attention_masks[train_idx], token_type_ids[train_idx], labels[train_idx]
    )
    val_data = TensorDataset(
        input_ids[val_idx], attention_masks[val_idx], token_type_ids[val_idx], labels[val_idx]
    )

    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    class_counts = np.bincount(labels.numpy())
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS
    )

    logs = []
    best_f1 = 0.0
    patience_counter = 0

    for epoch_i in range(EPOCHS):
        print(f"\n======== Epoch {epoch_i + 1} / {EPOCHS} ========")

        # === 학습 ===
        model.train()
        total_loss = 0
        for batch in train_loader:
            b_input_ids, b_input_mask, b_segment_ids, b_labels = (t.to(device) for t in batch)

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_input_mask)
            loss = label_smoothing_loss(outputs.logits, b_labels, smoothing=LABEL_SMOOTHING)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # === 평가 ===
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        preds, true_labels = [], []

        for batch in val_loader:
            b_input_ids, b_input_mask, b_segment_ids, b_labels = (t.to(device) for t in batch)
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_input_mask)
                loss = label_smoothing_loss(outputs.logits, b_labels, smoothing=LABEL_SMOOTHING)
                logits = outputs.logits

            eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            preds.append(logits)
            true_labels.append(label_ids)
            eval_accuracy += flat_accuracy(logits, label_ids)

        eval_loss /= len(val_loader)
        eval_accuracy /= len(val_loader)

        preds = np.concatenate(preds, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = true_labels.flatten()

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, preds_flat, average='macro', zero_division=0
        )
        conf_matrix = confusion_matrix(labels_flat, preds_flat)

        print(f"Validation Loss: {eval_loss:.4f}")
        print(f"Validation Accuracy: {eval_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        logs.append({
            'epoch': epoch_i + 1,
            'train_loss': avg_train_loss,
            'val_loss': eval_loss,
            'val_accuracy': eval_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            model.save_pretrained('../data/model_save/')
            tokenizer.save_pretrained('../data/model_save/')
            print("✅ Improved model saved.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    # 저장
    pd.DataFrame(logs).to_csv('../data/model_save/training_log.csv', index=False)
    with open('../data/model_save/training_config.json', 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "epochs": epoch_i + 1
        }, f, indent=2, ensure_ascii=False)

    print("training_log.csv and training_config.json saved.")

if __name__ == '__main__':
    main()
