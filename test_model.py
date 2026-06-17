from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import os

model = AutoModelForSequenceClassification.from_pretrained(os.path.join('models', 'DeepPavlov_rubert-base-cased'))
tokenizer = AutoTokenizer.from_pretrained(os.path.join('models', 'DeepPavlov_rubert-base-cased'))
labels_df = pd.read_csv(os.path.join('dataset', 'labels.csv'), sep=',')
device = torch.device("cpu")


def predict_intent(text, model, tokenizer):
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        # print(predictions)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()

    return predicted_class, confidence


if __name__ == '__main__':
    test_text = input('Вопрос клиента: ')
    pred, conf = predict_intent(test_text, model, tokenizer)
    print(f"📝 '{test_text}' -> {labels_df.loc[labels_df['номер темы'] == pred, 'тема'].values[0]} (уверенность: {conf:.2f})")
