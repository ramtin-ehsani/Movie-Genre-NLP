#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q transformers datasets')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train': 'drive/MyDrive/nlp_data/raw/train_data_for_bert.csv', 'test': 'drive/MyDrive/nlp_data/raw/test_data_for_bert.csv'})


# In[ ]:


dataset


# In[ ]:


dataset['train'][0]


# In[ ]:


labels = [label for label in dataset['train'].features.keys() if label not in ['preprocessed_plot']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels


# In[ ]:


from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["preprocessed_plot"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding


# In[ ]:


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)


# In[ ]:


encoded_dataset['train'][0].keys()


# In[ ]:


tokenizer.decode(encoded_dataset['train'][0]['input_ids'])


# In[ ]:


encoded_dataset['train'][0]['labels']


# In[ ]:


[id2label[i] for i, label in enumerate(encoded_dataset['train'][0]['labels']) if label == 1.0]


# In[ ]:


encoded_dataset.set_format("torch")


# In[ ]:


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)


# In[ ]:


from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)


# In[ ]:


from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


# In[ ]:


#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
outputs


# In[ ]:


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[ ]:


trainer.train()


# In[ ]:


pt_save_directory = "drive/MyDrive/nlp_data/pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)


# In[ ]:




