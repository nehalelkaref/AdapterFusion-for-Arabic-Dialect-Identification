import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def create_labels_dict(countries):
  labels_dict = {}
  for idx in range(len(countries)):
    labels_dict[countries[idx]] = idx
  return labels_dict

def create_labels(labels, labels_dict):
  new_labels = []
  for label in labels:
    new_label = labels_dict[label]
    new_labels.append(new_label)
  return new_labels

def encode_batch(batch, tokenizer):
  return tokenizer(batch["tweet"], max_length=300, truncation=True, padding="max_length")

def compute_metrics(pred_list):

    pred, labels = pred_list
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average = 'macro')
    precision = precision_score(y_true=labels, y_pred=pred, average = 'macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average = 'macro')
    
    score_dict={"accuracy": accuracy,
                "precision": precision,
                "recall": recall, 
                "f1": f1}
    return score_dict
