from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# model yükleme adımı 
model1_name = "unitary/toxic-bert"
tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
model1 = AutoModelForSequenceClassification.from_pretrained(model1_name)

model2_name = "Hate-speech-CNERG/dehatebert-mono-english"
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
model2 = AutoModelForSequenceClassification.from_pretrained(model2_name)

model3_name = "cardiffnlp/twitter-roberta-base-offensive"
tokenizer3 = AutoTokenizer.from_pretrained(model3_name)
model3 = AutoModelForSequenceClassification.from_pretrained(model3_name)

def get_twitter_roberta_offensive_score(text):
    text = text.replace('@user', '').replace('http', '')
    inputs = tokenizer3(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model3(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  # 'offensive' skoru

def score_toxicity(text):

    inputs1 = tokenizer1(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        probs1 = F.softmax(model1(**inputs1).logits, dim=-1)
    toxic_bert_score = probs1[0][1].item()

    # DehateBERT
    inputs2 = tokenizer2(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        probs2 = F.softmax(model2(**inputs2).logits, dim=-1)
    num_labels = probs2.shape[1]
    hate_score = probs2[0][1].item() if num_labels > 1 else 0.0
    offensive_score = probs2[0][2].item() if num_labels > 2 else 0.0
    dehate_bert_score = max(hate_score, offensive_score)

    # Twitter-Roberta
    twitter_offensive_score = get_twitter_roberta_offensive_score(text)

    # Finding the max score 
    model_score = max(toxic_bert_score, dehate_bert_score, twitter_offensive_score)

    if model_score > 0.3:
        explanation = (
            f"Model-based toxicity detected. "
            f"Toxic-BERT: {toxic_bert_score:.2f}, "
            f"DehateBERT: {dehate_bert_score:.2f}, "
            f"Twitter-Offensive: {twitter_offensive_score:.2f}"
        )
        return {
            "toxic_score": round(model_score, 3),
            "label": "Toxic",
            "explanation": explanation,
            "model_score": round(model_score, 3),
            "detected_toxic_phrases": [text]
        }
    else:
        return {
            "toxic_score": round(model_score, 3),
            "label": "Non-toxic",
            "explanation": "No toxicity detected.",
            "model_score": round(model_score, 3),
            "detected_toxic_phrases": []
        }
