
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


path = os.path.dirname(os.path.abspath("model_interface.py"))

model_checkpoints = [
    path + "/saved_models/codeparrot-small-conala-epoch-15-django-epoch-10",
    #path + "/saved_models/codeparrot-small-conala-epoch-29-django-epoch-10",
    #path + "/saved_models/codeparrot-small-conala-epoch-15",
    #path + "/saved_models/codeparrot-small-conala-epoch-29",
    #path + "/saved_models/codeparrot-conala-epoch-15-django-epoch-2",
]


model_checkpoint = model_checkpoints[0]
tokenizer_checkpoint = model_checkpoints[0]

model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = tokenizer.eos_token


while True:

    prompt = input("prompt: ")

    if prompt == "exit":
        exit()
    prompt = "'''" + prompt + "'''\n"

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    sample_outputs = model.generate(
            input_ids,
            do_sample=True, 
            max_length= len(input_ids[0]) + 15, 
            top_k=50, 
            top_p=0.95,
            #num_beams = 20,
            num_return_sequences=3,
            temperature = 0.2,
            remove_invalid_values=True,

    )      
    
    for i, output in enumerate(sample_outputs):

        generated_full_text= tokenizer.decode(sample_outputs[i], skip_special_tokens=True)
        generated_snippet_text = generated_full_text[len(prompt):]
        print(generated_snippet_text)
