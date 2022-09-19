
"""
Developed by Tomas Pimentel Zilhao Pinto e Borges, 201372847
COMP3931 Individual Project
""" 

import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from datasets import load_metric
import datasets
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import AutoModelForSequenceClassification,AutoModel, AutoModelWithLMHead, GPT2LMHeadModel, AutoModelForCausalLM
from transformers import TrainingArguments,Trainer
from transformers import pipeline
import transformers

from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import copy


transformers.logging.set_verbosity_error()


path = os.path.dirname(os.path.abspath("eval_nll.py"))


model_checkpoints = [
    path + "/saved_models/codeparrot-small-conala-epoch-15-django-epoch-10",
    "lvwerra/codeparrot-small",
    "microsoft/CodeGPT-small-py-adaptedGPT2",
    "microsoft/CodeGPT-small-py",
    "EleutherAI/gpt-neo-125M"
]

tokenizer_checkpoint = path + "/saved_models/codeparrot-small-conala-epoch-15-django-epoch-10" 

model = AutoModelForCausalLM.from_pretrained(model_checkpoints[0])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = tokenizer.eos_token



base_url = path + "/saved_datasets/"
django_dataset = load_dataset('json', data_files={
    'test':base_url+ 'django_test.jsonl'
})

#django_dataset = load_dataset("AhmedSSoliman/CoNaLa") CoNaLa calculations



def tokenize_function(example):

    tokenized1 = tokenizer( np.char.add( ["'''"] *len(example["intent"]),
        np.char.add(example["intent"], ["'''\n"] *len(example["intent"])) 
        ).tolist(), 
        example["snippet"], truncation=True) 
    

    tokenized1["labels"] = copy.deepcopy(tokenized1["input_ids"])

    tokenized2 = tokenizer(np.char.add( ["'''"] *len(example["intent"]),
        np.char.add(example["intent"], ["'''\n"] *len(example["intent"])) 
        ).tolist(), 
        truncation=True)

    for i in range(0,len(tokenized1["input_ids"])): #ignoring loss on prompt ids
        tokenized1["labels"][i][0:len(tokenized2["input_ids"][i])] = [-100]*len(tokenized2["input_ids"][i])


    return tokenized1



django_tokenized_dataset = django_dataset.map(tokenize_function, batched=True) 
django_tokenized_dataset = django_tokenized_dataset.remove_columns(["intent", "snippet"]) 
django_tokenized_dataset.set_format("torch")


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

django_test_dataloader = DataLoader(
    django_tokenized_dataset["test"],  batch_size=1, collate_fn=data_collator,
) 


# inspect a batch to check that there is no mistake 
for batch in django_test_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
print(batch.items())
# transformers models will return the loss when labels are provided 
outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"], use_cache=False) 
print(outputs.loss, outputs.logits.shape)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

valid_progress_bar = tqdm(range(len(django_test_dataloader)))

django_total_valid_loss = 0.0
django_avg_valid_loss = 0.0

model.eval() 
for batch_id, batch in enumerate(django_test_dataloader):

    
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"], use_cache=False) 

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    loss = outputs.loss
    django_total_valid_loss += loss.item()
    valid_progress_bar.update(1)


django_avg_valid_loss = django_total_valid_loss/len(django_test_dataloader)

print("Average Negative Log Likelihood for each token for each example in dataset")
print(django_avg_valid_loss)
