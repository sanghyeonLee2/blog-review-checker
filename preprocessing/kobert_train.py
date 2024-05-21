import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def convert_to_bert_inputs(text_list, max_len, tokenizer):
    input_ids, attention_masks, token_type_ids = [], [], []

    for text in text_list:
        text = "" if pd.isnull(text) else text
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
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
    # Config
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 4
    MODEL_NAME = 'monologg/kobert'

    # Load data
    df = pd.read_csv('../data/processed_output.csv', encoding='utf-8-sig')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    input_ids, attention_masks, token_type_ids = convert_to_bert_inputs(
        df['combined_text'].values, MAX_LEN, tokenizer
    )
    labels = torch.tensor(df['blog_is_promotional'].values, dtype=torch.long)

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

    train_inputs = input_ids[train_idx]
    val_inputs = input_ids[val_idx]
    train_masks = attention_masks[train_idx]
    val_masks = attention_masks[val_idx]
    train_types = token_type_ids[train_idx]
    val_types = token_type_ids[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_data = TensorDataset(train_inputs, train_masks, train_types, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_types, val_labels)

    # Windows 환경: num_workers=0
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    class_counts = np.bincount(labels.numpy())
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch_i in range(EPOCHS):
        print(f'======== Epoch {epoch_i + 1} / {EPOCHS} ========')

        model.train()
        total_loss = 0
        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_segment_ids, b_labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_input_mask)
            loss = loss_fn(outputs.logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"Average training loss: {total_loss / len(train_dataloader):.2f}")

        # Evaluation
        model.eval()
        eval_loss = 0
        eval_accuracy = 0
        preds, true_labels = [], []

        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_segment_ids, b_labels = tuple(t.to(device) for t in batch)
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

        eval_loss /= len(val_dataloader)
        eval_accuracy /= len(val_dataloader)
        print(f"Validation Loss: {eval_loss:.2f}")
        print(f"Validation Accuracy: {eval_accuracy:.2f}")

        preds = np.concatenate(preds, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = true_labels.flatten()

        precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='binary')
        conf_matrix = confusion_matrix(labels_flat, preds_flat)

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

    output_dir = '../data/model_save/'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model and config saved to {output_dir}")


if __name__ == '__main__':
    main()