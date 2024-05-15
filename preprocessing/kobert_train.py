import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from imblearn.over_sampling import SMOTE

# 전처리된 데이터 불러오기
df = pd.read_csv('../data/processed_output.csv', encoding='cp949')

# KoBERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

# 최대 시퀀스 길이 설정
MAX_LEN = 128

# 텍스트 데이터를 KoBERT 입력 형식으로 변환하는 함수
def convert_to_kobert_inputs(text_list, max_len):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for text in text_list:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 스페셜 토큰 추가
            max_length=max_len,  # 최대 길이 설정
            padding='max_length',  # 패딩
            return_attention_mask=True,  # 어텐션 마스크 생성
            truncation=True
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])  # 세그먼트 인덱스

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(token_type_ids)

# 'combined_text' 열을 KoBERT 입력 형식으로 변환
input_ids, attention_masks, token_type_ids = convert_to_kobert_inputs(df['combined_text'].values, MAX_LEN)
labels = torch.tensor(df['blog_is_promotional'].values)

# SMOTE 적용
smote = SMOTE(random_state=42)
input_ids_resampled, labels_resampled = smote.fit_resample(input_ids.numpy(), labels.numpy())
attention_masks_resampled, _ = smote.fit_resample(attention_masks.numpy(), labels.numpy())
token_type_ids_resampled, _ = smote.fit_resample(token_type_ids.numpy(), labels.numpy())

# 텐서로 변환
input_ids_resampled = torch.tensor(input_ids_resampled)
labels_resampled = torch.tensor(labels_resampled)
attention_masks_resampled = torch.tensor(attention_masks_resampled)
token_type_ids_resampled = torch.tensor(token_type_ids_resampled)

# 학습 데이터와 테스트 데이터 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids_resampled, labels_resampled, test_size=0.2, random_state=42)
train_masks, validation_masks, _, _ = train_test_split(attention_masks_resampled, labels_resampled, test_size=0.2, random_state=42)
train_type_ids, validation_type_ids, _, _ = train_test_split(token_type_ids_resampled, labels_resampled, test_size=0.2, random_state=42)

# 데이터 로더 생성
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_type_ids, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=4)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_type_ids, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, num_workers=4)

# KoBERT 모델 로드
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 옵티마이저와 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)  # 학습률을 낮추어 설정
epochs = 4
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)

# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 학습 루프
for epoch_i in range(0, epochs):
    print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
    
    # 학습
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_segment_ids = b_segment_ids.to(device)
        b_labels = b_labels.to(device)
        
        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=b_segment_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.2f}")

    # 검증
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    nb_eval_steps = 0
    preds, true_labels = [], []

    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_segment_ids = b_segment_ids.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=b_segment_ids,
                            attention_mask=b_input_mask)
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        preds.append(logits)
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_steps
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

# 모델 저장
output_dir = './model_save/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model saved to %s" % output_dir)
