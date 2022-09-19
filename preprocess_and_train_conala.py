
"""
Developed by Tomas Pimentel Zilhao Pinto e Borges, 201372847
COMP3931 Individual Project
""" 

import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from datasets import load_metric
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

### Utility to monitor GPU and Memory usage during training
import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        
# Instantiate monitor with a 10-second delay between updates
# monitor = Monitor(10)

# Train, etc.

# Close monitor
# monitor.stop()
###

path = os.path.dirname(os.path.abspath("preprocess_and_train_conala.py"))


# LOAD MODEL AND TOKENIZER
model_checkpoint = "lvwerra/codeparrot-small" #remove small when training bigger model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token


# LOAD DATASET
raw_datasets = load_dataset("AhmedSSoliman/CoNaLa") 


# PRE PROCESSING DATASET
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




tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) 
tokenized_datasets = tokenized_datasets.remove_columns(["intent", "snippet"]) 
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=1, collate_fn=data_collator
)
vali_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=1, collate_fn=data_collator
)


# inspect a batch to check that there is no mistake 
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
print(batch.items())
# transformers models will return the loss when labels are provided 
outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"], use_cache=False) 
print(outputs.loss, outputs.logits.shape)

# TRAINING 

num_epochs = 30
num_training_steps = num_epochs * len(train_dataloader)

# optimizer
optimizer = AdamW(
    model.parameters(), 
    lr=1e-3, 
    betas = [0.9,0.95],
    weight_decay= 0.1,
    eps = 1e-8
    )

# learning rate optimizer
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

# where to save the model and loss outputs
output_dir = path + "/codeparrot-conala-small" #remove small when training bigger model
f1 = open(output_dir  + "-training-loss.txt", "w")
f2 = open(output_dir  + "-validation-loss.txt", "w")
f1.write("epoch|step|loss|batch['input_ids'].shape\n")
f2.write("epoch|step|loss|batch['input_ids'].shape\n")

# progress bars
train_progress_bar = tqdm(range(num_training_steps))
valid_progress_bar = tqdm(range(num_epochs*len(vali_dataloader)))

# gradient accumulation
accum_iter = 8

# training (and validation) loop 

# monitor = Monitor(10)
min_valid_loss = np.inf
for epoch in range(num_epochs):

    epoch_train_loss = 0.0
    epoch_valid_loss = 0.0

    # tells your model that you are training the model. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.
    # It is somewhat intuitive to expect train function to train model but it does not do that. It just sets the mode.
    model.train()  

    #training loop
    for batch_id, batch in enumerate(train_dataloader):

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"], use_cache=False) 
        loss = outputs.loss
        loss.backward()

        # gradient accumulation:
        # The gradients are computed when we call loss.backward() and are stored by PyTorch until we call optimizer.zero_grad()
        # Therefore, we just need to move the weight update performed in optimizer.step() and the gradient reset under the if condition; also moving the learning rate update
        if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(train_dataloader)):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        train_progress_bar.update(1)
        epoch_train_loss += loss.item()
        f1.write(str(epoch) + "|" + str(batch_id) + "|"+ str(loss.item()) + "|" + str(batch["input_ids"].shape) + "\n")

    # validation loop
    model.eval() 
    for batch in vali_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"], use_cache=False) 

        loss = outputs.loss
        epoch_valid_loss += loss.item()
        valid_progress_bar.update(1)
        f2.write(str(epoch) + "|" + str(batch_id) + "|"+ str(loss.item()) + "|" +  str(batch["input_ids"].shape) + "\n")

    average_train_loss = epoch_train_loss / len(train_dataloader)
    average_vali_loss = epoch_valid_loss / len(vali_dataloader)
    f1.write(f'Epoch {epoch+1} \t\t (average) Training Loss: {average_train_loss} \n')
    f2.write(f'Epoch {epoch+1} \t\t (average) Validation Loss: {average_vali_loss} \n')

    if min_valid_loss > average_vali_loss:
        f2.write(f'(average) Validation Loss Decreased({min_valid_loss:.6f}--->{average_vali_loss:.6f})\n')
        min_valid_loss = average_vali_loss
    else:
        f2.write(f'(average) Validation Loss Increased!({min_valid_loss:.6f}--->{average_vali_loss:.6f})\n')
        # if validation loss increases, we save the model
        print("Saving model after epoch" + str(epoch))
        model.save_pretrained(output_dir + "-epoch-" + str(epoch) + "/")
        tokenizer.save_pretrained(output_dir + "-epoch-" + str(epoch) + "/")
        torch.save(optimizer.state_dict(), output_dir + "-epoch-" + str(epoch) + "/" + 'optimizer.pt')
        torch.save(lr_scheduler.state_dict(), output_dir + "-epoch-" + str(epoch) + "/" 'scheduler.pt')
        #break

    # periodically saving checkpoint
    if epoch % 5 == 0 and epoch > 0:
        print("Saving model after epoch " + str(epoch))
        model.save_pretrained(output_dir + "-epoch-" + str(epoch) + "/")
        tokenizer.save_pretrained(output_dir + "-epoch-" + str(epoch) + "/")
        torch.save(optimizer.state_dict(), output_dir + "-epoch-" + str(epoch) + "/" + 'optimizer.pt')
        torch.save(lr_scheduler.state_dict(), output_dir + "-epoch-" + str(epoch) + "/" 'scheduler.pt')


f1.close()
f2.close()
# monitor.stop()

# SAVING FINAL MODEL AND TOKENIZER
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

