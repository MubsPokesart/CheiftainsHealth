# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas
# %pip install accelerate -U
# %pip install accelerate -U

import os
import evaluate
import torch as pt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, RobertaTokenizerFast, RobertaForSequenceClassification


id2label = {0: 'non-urgent', 1: 'urgent'}
    
def read_clean_triage_data(path):
    # General Data Cleaning
    df = pd.read_csv(path)
    
    df[df.isnull().T.any().T]
    df['question'] = df['question'].astype('string')
    df['triage'].value_counts()
    
    
    df['triage'] = df['triage'].apply(lambda x: 0 if x == 'non-urgent' else 1)
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def init_trainer(model, epochs = 2, out_dir = "./results"):

    training_args = TrainingArguments(
        output_dir = out_dir,
        num_train_epochs = epochs,
        evaluation_strategy='steps',
        logging_dir='./logs',
        logging_steps = 10,
        logging_first_step = True,
        warmup_steps = 500,
        weight_decay=0.01,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 64,
        gradient_accumulation_steps = 8,
        gradient_checkpointing=True,
        # fp16 = True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    return trainer



def prepare_data_training(df):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_texts, val_texts, train_labels, val_labels = train_test_split(df["question"], df["triage"], test_size=0.2, shuffle=True)
    
    train_texts = train_texts.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True).array
    val_labels = val_labels.reset_index(drop=True).array

    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

    train_data = QuestionsDataset(train_encodings, train_labels)
    val_data = QuestionsDataset(val_encodings, val_labels)

    metric = evaluate.load("accuracy")

    return model, train_data, val_data, metric


def train_model(model, train_data, val_data, metric, epochs = 2, out_dir = "./results"):
    trainer = init_trainer(model, epochs, out_dir)
    trainer.train()
    trainer.save_model()
    trainer.evaluate()

    return trainer

if __name__ == "__main__":
    df = read_clean_triage_data("triage_data.csv")
    model, train_data, val_data, metric = prepare_data_training(df)
    
    trainer = train_model(model, train_data, val_data, metric, 2, "./results/train1") # this model is DistilBert
    trainer.save_model()
    trainer.evaluate()

# Section 2: Prepare Data for Training


# Artifacts from attempts of different models 
# from transformers import ElectraTokenizer, ElectraForSequenceClassification
# model_name = "google/electra-small-discriminator"
# tokenizer = ElectraTokenizer.from_pretrained(model_name)
# model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=2)

model_name = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
train_texts, val_texts, train_labels, val_labels = train_test_split(df["question"], df["triage"], test_size=0.2, shuffle=True)

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)



class QuestionsDataset(pt.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: pt.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = pt.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_data = QuestionsDataset(train_encodings, train_labels)
val_data = QuestionsDataset(val_encodings, val_labels)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def init_trainer(model, epochs = 2, out_dir = "./results"):

    training_args = TrainingArguments(
        output_dir = out_dir,
        num_train_epochs = epochs,
        evaluation_strategy='steps',
        logging_dir='./logs',
        logging_steps = 10,
        logging_first_step = True,
        warmup_steps = 500,
        weight_decay=0.01,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 64,
        gradient_accumulation_steps = 8,
        gradient_checkpointing=True,
        # fp16 = True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    return trainer

# Model Fine-Tuning
trainer1 = init_trainer(model, 2, "./results/train1") # this model is DistilBert
trainer1.train()
trainer1.save_model()
trainer1.evaluate()

trainer2 = init_trainer(model, 2, "./results/train2") # this model is Roberta
trainer2.train()
trainer2.save_model()
trainer2.evaluate()

## AI Testing
# Commented out IPython magic to ensure Python compatibility.
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch as pt
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
id2label = {0: 'non-urgent', 1: 'urgent'}

model = DistilBertForSequenceClassification.from_pretrained("./results/train1", local_files_only=True)

def test_model(message):
  encodings = tokenizer(message, truncation=True, padding=True, return_tensors="pt")
  output = model(**encodings)
  result = pt.argmax(output.logits).item()
  return id2label[result]

test_model("Please help, what should I do? My father fell and he can't get up.")
test_model("What does it mean if my son gets sick every time we go to swimming lessons?")
test_model("Please help!! My baby is crying urgently!") # bad parent, not urgent
test_model("What do I do? My mommy isn't waking up I'm poking her and she won't move.") # mom could be dead, urgent

"""AI generated examples"""

examples_urgent = (
    "I'm experiencing severe chest pain and shortness of breath. I think I may be having a heart attack.",
    "My child has a high fever of 104°F and has been vomiting for the past 6 hours. We need to be seen immediately.",
    "I just had a bad fall and I think my arm might be broken. I'm in a lot of pain and need to come in right away.",
    "I'm pregnant and I'm bleeding heavily. I'm very concerned and need to speak with a doctor as soon as possible.",
    "I've been experiencing severe abdominal pain and diarrhea for the past 24 hours. I'm feeling extremely weak and dehydrated.")

for example in examples_urgent:
    print(test_model(example))

examples_nonurgent = (
    "Hello, I was wondering if I could schedule an appointment for a routine check-up next week?",
    "I need to refill my prescription for blood pressure medication. Could you please send a refill request to my pharmacy?",
    "I have a question about the side effects of the medication I'm currently taking. Is there a good time I could speak with a nurse?",
    "I'd like to get a flu shot this year. Can you let me know when the clinic has the flu vaccine available?",
    "I need to update my insurance information on file. Could you please provide me with the necessary forms?",)

for example in examples_nonurgent:
    print(test_model(example))

examples_trick = (
    "Please help!! My baby is crying urgently!",
    "I have a terrible headache, it's an emergency!",
    "I need to schedule a very urgent appointment for a papercut.",
    "It's an absolute crisis, I broke a nail!",
    "This is an emergency! I ran out of my daily multivitamin.",
    "I'm having an urgent problem with a hangnail, I need to be seen right away!",
    "Emergency! I stubbed my toe and it's a little bruised.",
    "Urgent request: I need a refill on my dandruff shampoo.",
    "This is an urgent matter - I need the phone number for a dog groomer.",
    "I'm having an emergency allergic reaction to pollen. Just kidding, I'm fine!"
)

for example in examples_trick:
    print(test_model(example))

examples_false_neg = (
    "There's some blood in my urine, but I did just start a new medication so that might be causing it.",
    "I'm feeling a little dizzy and lightheaded, but I'm sure it's nothing serious.",
    "I have a sharp pain in my lower abdomen, but it comes and goes so I'm not too worried.",
    "I found a lump in my breast, but I'm sure it's just a harmless cyst.",
    "I was playing basketball and took a hard fall on my knee. It's swollen but I can still walk on it.",
        "My son fell off the monkey bars at the park and hit his head pretty hard. He says he feels fine though.",
    "I've had this really bad headache for 3 days straight now. Could be a migraine or something?",
    "I'm a type 1 diabetic and my blood sugar has been over 300 for most of the day. Should I be concerned?",
    "My Dad is 75 and has been having trouble remembering things lately. Is that just normal aging?",
    "I'm 32 weeks pregnant and haven't felt the baby move much today. Could you let me know if that's ok?",
    "I've had some tightness in my chest and arm numbness for a few hours now. Maybe I slept weird?",
    "My 3-year-old daughter spiked a fever of 103 this morning but she's acting totally normal otherwise.",
    "There's blood in my stool, maybe it's just hemorrhoids? I've had them before so hopefully that's all it is."
)

for example in examples_false_neg:
    print(test_model(example))