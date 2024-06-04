from kobert_transformers import get_tokenizer
import torch
from transformers import BertForSequenceClassification
from app.services.preprocessing import convert_to_kobert_inputs

# KoBERT 토크나이저 및 모델 불러오기
tokenizer = get_tokenizer()
model = BertForSequenceClassification.from_pretrained('../data/model_save')
model.eval()

# 단일 블로그 텍스트 예측 함수
def predict_is_promotional(cleaned_text):
    input_ids, attention_mask, token_type_ids = convert_to_kobert_inputs(cleaned_text, 128, tokenizer)

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    is_promotional = int(probs.argmax())
    probability = float(probs[is_promotional])
    return is_promotional, probability

