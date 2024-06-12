# -*- coding: utf-8 -*-
import numpy as np
from datasets import load_dataset, Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import json
import torch
import torch.nn.functional as F
import wandb

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75,gamma=3.0,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        valid_mask = (targets != self.ignore_index)
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha*(1-pt)**self.gamma*ce_loss

        return focal_loss.mean()

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits=outputs.get('logits')

        loss = self.focal_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss



if torch.cuda.is_available():
    device = torch.device("cuda:0")

label_list = ["O","B-PER","I-PER","B-LOC",
    "I-LOC",
    "B-CORP",
    "I-CORP",
    "B-GRP",
    "I-GRP",
    "B-PROD",
    "I-PROD",
    "B-CW",
    "I-CW"]
label_encoding_dict = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-CORP": 5,
    "I-CORP": 6,
    "B-GRP": 7,
    "I-GRP": 8,
    "B-PROD": 9,
    "I-PROD": 10,
    "B-CW": 11,
    "I-CW": 12,
}
with open('./results.txt', 'w'):
    pass
def read_json(path):
    # Open the JSON file for reading
    with open(path, 'r', encoding='utf-8') as file:
        # Parse the JSON file
        data = json.load(file)

    labels = []

    for inst in data:
        l_list = []
        for l in inst['label']:
            l_list.append(label_encoding_dict[l])
        labels.append(l_list)

    converted_data = {
        'tokens': [d['text'] for d in data],
        'labels': labels
    }
    dataset = Dataset.from_dict(converted_data)
    return dataset

# multiconer
model_checkpoint = 'xlm-roberta-base'
# Define the list of NER labels and their corresponding numerical encoding
batch_size = 16

# Function to process the raw dataset, remove unnecessary columns, and rename the ner_tags column to labels
def get_dataset(path_train, path_dev, path_test):
    train_dataset = read_json(path_train)
    validation_dataset = read_json(path_dev)
    test_dataset = read_json(path_test)
    return train_dataset, validation_dataset, test_dataset

# Function to tokenize input text and align the NER labels with the tokenized text
def tokenize_and_align_labels(examples, max_length=512):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_checkpoint)
    tokenized_samples = tokenizer.batch_encode_plus(examples["tokens"], is_split_into_words=True, max_length=max_length,truncation=True)
    total_adjusted_labels = []

    for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = examples["labels"][k]
        i = -1
        adjusted_label_ids = []

        for wid in word_ids_list:
            if(wid is None):
                adjusted_label_ids.append(-100)
            elif(wid!=prev_wid):
                i = i + 1
                adjusted_label_ids.append(existing_label_ids[i])
                prev_wid = wid
            else:
                label_name = label_list[existing_label_ids[i]]
                adjusted_label_ids.append(existing_label_ids[i])

        total_adjusted_labels.append(adjusted_label_ids)
    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples

# Function to compute evaluation metrics (precision, recall, F1 score, and accuracy)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    metric = load_metric("seqeval")

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

def main():
    shots = [5,10,20,50]
    lans = ['BN-Bangla', 'DE-German', 'EN-English', 'ES-Spanish', 'FA-Farsi', 'HI-Hindi', 'KO-Korean', 'NL-Dutch', 'RU-Russian', 'TR-Turkish', 'ZH-Chinese']
    for shot in shots:
        for lan in lans:
            current = str(shot)+' shots for ' + lan + ':\n'
            print(current)
            path_dev = './data/' + lan + '/dev.json'
            path_test = './data/' + lan + '/test.json'
            validation_dataset = read_json(path_dev)
            test_dataset = read_json(path_test).select(range(1000))
            # Tokenize and align labels for the train, validation, and test datasets
            validation_tokenized_datasets = validation_dataset.map(tokenize_and_align_labels, batched=True)
            test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)
            path_train = './data/' + lan + '/' + str(shot) + 'shots.json'
            train_dataset = read_json(path_train)
            train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
            model = XLMRobertaForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_checkpoint)

            # Set training arguments and hyperparameters
            args = TrainingArguments(
                "test-ner",
                evaluation_strategy = "epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=3,
                weight_decay=2e-4,
                metric_for_best_model="f1",  # Choose the best model based on the F1 score
            )

            # Initialize the data collator for token classification
            data_collator = DataCollatorForTokenClassification(tokenizer)
            metric = load_metric("seqeval")

            # Instantiate the Trainer class with the processed datasets, data collator, tokenizer, and compute_metrics function
            trainer = CustomTrainer(
                model,
                args,
                train_dataset=train_tokenized_datasets,
                eval_dataset=validation_tokenized_datasets,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            # Train the model
            trainer.train()

            # Save the best model
            #trainer.save_model('/content/drive/MyDrive/templateNER-main/saved_models/bert-conll-ner2.model')

            # Use the best model for prediction
            predictions, labels, _ = trainer.predict(test_tokenized_datasets)
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)

            with open('./results.txt', 'a') as file:
                file.write(current)
                file.write(str(results))
                file.write('\n')

if __name__ == '__main__':
    main()