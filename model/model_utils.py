import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from data.preprocessing import preprocess


# Load the saved model and tokenizer
model_path = "model/fine_tuned_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def classify_text(text):
    text = preprocess(text)
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label_id = torch.argmax(logits, dim=1).item()

    return predicted_label_id


def classify_text_plus(text):
    text =  preprocess(text)
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)  # Compute softmax to get probabilities
    predicted_label_id = torch.argmax(logits, dim=1).item()
    predicted_probabilities = probabilities[0].tolist()  # Convert probabilities to a list

    return predicted_label_id, predicted_probabilities


def get_label(code):
    actions = ['greeting', 'cancel_order', 'change_order', 'change_shipping_address', 'check_cancellation_fee', 'check_invoice', 'check_payment_methods', 'check_refund_policy', 'complaint', 'contact_customer_service', 'contact_human_agent', 'create_account', 'delete_account', 'delivery_options', 'delivery_period', 'edit_account', 'get_invoice', 'get_refund', 'newsletter_subscription', 'payment_issue', 'place_order', 'recover_password', 'registration_problems', 'review', 'set_up_shipping_address', 'switch_account', 'track_order', 'track_refund']
    indices = [17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    labels_dict = dict(zip(indices, actions))

    return labels_dict[code]
