import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from imblearn.over_sampling import SMOTE

def convert_to_kobert_inputs(text_list, max_len, tokenizer):
    input_ids, attention_masks, token_type_ids = [], [], []

    for text in text_list:
        text = "" if pd.isnull(text) else text  # 안전한 텍스트 처리 추가
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
        torch.tensor(input_ids),
        torch.tensor(attention_masks),
        torch.tensor(token_type_ids)
    )

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def main():
    df = pd.read_csv('../data/processed_output.csv', encoding='utf-8-sig')

    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    MAX_LEN = 128

    input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(df['combined_text'].values, MAX_LEN, tokenizer)
    labels = torch.tensor(df['blog_is_promotional'].values)

    # SMOTE 적용
    smote = SMOTE(random_state=42)
    input_ids_resampled, labels_resampled = smote.fit_resample(input_ids.numpy(), labels.numpy())
    attention_masks_resampled, _ = smote.fit_resample(attention_masks.numpy(), labels.numpy())
    token_type_ids_resampled, _ = smote.fit_resample(token_type_ids.numpy(), labels.numpy())

    # 텐서 변환
    input_ids_resampled = torch.tensor(input_ids_resampled)
    labels_resampled = torch.tensor(labels_resampled)
    attention_masks_resampled = torch.tensor(attention_masks_resampled)
    token_type_ids_resampled = torch.tensor(token_type_ids_resampled)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids_resampled, labels_resampled, test_size=0.2, random_state=42)
    train_masks, val_masks, _, _ = train_test_split(attention_masks_resampled, labels_resampled, test_size=0.2, random_state=42)
    train_types, val_types, _, _ = train_test_split(token_type_ids_resampled, labels_resampled, test_size=0.2, random_state=42)

    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_types, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_types, val_labels)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size, num_workers=0)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size, num_workers=0)

    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch_i in range(epochs):
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')

        # Train
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_segment_ids, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"Average training loss: {total_loss / len(train_dataloader):.2f}")

        # Validation
        model.eval()
        eval_accuracy = 0
        nb_eval_steps = 0
        preds, true_labels = [], []

        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_segment_ids, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            preds.append(logits)
            true_labels.append(label_ids)
            eval_accuracy += flat_accuracy(logits, label_ids)
            nb_eval_steps += 1

        eval_accuracy /= nb_eval_steps
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

    # Save
    output_dir = '../data/model_save/'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == '__main__':
    main()