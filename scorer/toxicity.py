import os
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Dosya yolu: toxicity.py'nin bulunduğu klasör
base_dir = os.path.dirname(os.path.abspath(__file__))
toxics_json_path = os.path.join(base_dir, "toxics.json")

# JSON'dan toksik kelimeleri oku (word + weight yapısıyla)
with open(toxics_json_path, "r", encoding="utf-8") as f:
    toxic_keywords = json.load(f)["toxic_keywords"]

# MODELLER
model1_name = "unitary/toxic-bert"
tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
model1 = AutoModelForSequenceClassification.from_pretrained(model1_name)

model2_name = "Hate-speech-CNERG/dehatebert-mono-english"
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
model2 = AutoModelForSequenceClassification.from_pretrained(model2_name)

# Metindeki TÜM toksik kelimeleri ve ağırlıklarını bulur
def find_all_toxic_keywords(text):
    text_lower = text.lower()
    found = []
    for kw in toxic_keywords:
        word = kw["word"]
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            found.append(kw)  # dict olarak ekle (word + weight)
    return found

def score_toxicity(text):
    # Model 1: Toxic-BERT
    inputs1 = tokenizer1(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs1 = model1(**inputs1)
        probs1 = F.softmax(outputs1.logits, dim=-1)
    toxic_bert_score = probs1[0][1].item()

    # Model 2: DehateBERT
    inputs2 = tokenizer2(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs2 = model2(**inputs2)
        probs2 = F.softmax(outputs2.logits, dim=-1)
    num_labels = probs2.shape[1]
    hate_score = probs2[0][1].item() if num_labels > 1 else 0.0
    offensive_score = probs2[0][2].item() if num_labels > 2 else 0.0
    dehate_bert_score = max(hate_score, offensive_score)

    # Keyword bazlı
    toxic_words = find_all_toxic_keywords(text)
    word_list = [w["word"] for w in toxic_words]
    weight_list = [w["weight"] for w in toxic_words]
    keyword_score_max = max(weight_list) if weight_list else 0.0

    model_score = max(toxic_bert_score, dehate_bert_score)

    if toxic_words:
        # Eğer keyword ve model skorları varsa, küçük olanı dikkate al
        final_score = min(model_score, keyword_score_max)
        explanation = (
            f"Toxic word(s) detected: {', '.join(word_list)} (keyword filter, weights: {', '.join(str(w) for w in weight_list)}). "
            f"Model score: {model_score:.2f}. Used minimum for final score."
        )
        label = "Toxic" if final_score > 0.3 else "Non-toxic"
        return {
            "toxic_score": round(final_score, 3),
            "label": label,
            "explanation": explanation,
            "toxic_words": word_list,
            "weights": weight_list,
            "model_score": round(model_score, 3),
            "keyword_score": keyword_score_max,
            "source": "min(model,keyword)"
        }
    else:
        if model_score > 0.3:
            explanation = f"Model-based toxicity detected. Toxic-BERT: {toxic_bert_score:.2f}, DehateBERT: {dehate_bert_score:.2f}"
            return {
                "toxic_score": round(model_score, 3),
                "label": "Toxic",
                "explanation": explanation,
                "toxic_words": [],
                "weights": [],
                "model_score": round(model_score, 3),
                "keyword_score": 0,
                "source": "model"
            }
        else:
            return {
                "toxic_score": round(model_score, 3),
                "label": "Non-toxic",
                "explanation": "No toxicity detected.",
                "toxic_words": [],
                "weights": [],
                "model_score": round(model_score, 3),
                "keyword_score": 0,
                "source": "none"
            }

