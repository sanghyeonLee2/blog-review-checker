import os
import json
import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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

def main():
    set_seed(42)

    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 4
    MODEL_NAME = 'monologg/kobert'

    df = pd.read_csv('../data/processed_output.csv', encoding='utf-8-sig')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(
        df['combined_text'].values, MAX_LEN, tokenizer
    )
    labels = torch.tensor(df['blog_is_promotional'].values, dtype=torch.long)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
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
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS
    )

    logs = []

    for epoch_i in range(EPOCHS):
        print(f'======== Epoch {epoch_i + 1} / {EPOCHS} ========')

        model.train()
        total_loss = 0
        for batch in train_loader:
            b_input_ids, b_input_mask, b_segment_ids, b_labels = (t.to(device) for t in batch)

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_input_mask)
            loss = loss_fn(outputs.logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.2f}")

        model.eval()
        eval_loss = 0
        eval_accuracy = 0
        preds, true_labels = [], []

        for batch in val_loader:
            b_input_ids, b_input_mask, b_segment_ids, b_labels = (t.to(device) for t in batch)
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_input_mask)
                loss = loss_fn(outputs.logits, b_labels)
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
            labels_flat, preds_flat, average='binary', zero_division=0
        )
        conf_matrix = confusion_matrix(labels_flat, preds_flat)

        print(f"Validation Loss: {eval_loss:.2f}")
        print(f"Validation Accuracy: {eval_accuracy:.2f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
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

    output_dir = '../data/model_save/'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        }, f, indent=2)

    pd.DataFrame(logs).to_csv(os.path.join(output_dir, "training_log.csv"), index=False)
    print("training_config.json and training_log.csv saved.")

if __name__ == '__main__':
    main()