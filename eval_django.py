
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


path = os.path.dirname(os.path.abspath("eval_django.py"))



model_checkpoints = [
    path + "/saved_models/codeparrot-small-conala-epoch-15-django-epoch-10", #0
    "lvwerra/codeparrot-small", #1
    "microsoft/CodeGPT-small-py-adaptedGPT2", #2
    "microsoft/CodeGPT-small-py", #3
    "EleutherAI/gpt-neo-125M", #4
]

tokenizer_checkpoint = path + "/saved_models/codeparrot-small-conala-epoch-15-django-epoch-10" 
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

base_url = path + "/saved_datasets/"
django_dataset = load_dataset('json', data_files={
    'test':base_url+ 'django_test.jsonl'
})


def tokenize_intents(example):

    tokenized1 = tokenizer( np.char.add( ["'''"] *len(example["intent"]),
        np.char.add(example["intent"], ["'''\n"] *len(example["intent"])) 
        ).tolist(), 
        truncation=True, max_length=512) 

    return tokenized1

def tokenize_snippets(example):

    tokenized1 = tokenizer( example["snippet"], truncation=True, max_length=512) 

    return tokenized1


django_tokenized_intents = django_dataset.map(tokenize_intents, batched=True) 
django_tokenized_intents = django_tokenized_intents.remove_columns(["intent", "snippet"]) 
django_tokenized_intents.set_format("torch")

django_tokenized_snippets = django_dataset.map(tokenize_snippets, batched=True) 
django_tokenized_snippets = django_tokenized_snippets.remove_columns(["intent", "snippet"]) 
django_tokenized_snippets.set_format("torch")


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


django_test_intents_dataloader = DataLoader(
    django_tokenized_intents["test"],  batch_size=1, collate_fn=data_collator,
) 

django_test_snippets_dataloader = DataLoader(
    django_tokenized_snippets["test"],  batch_size=1, collate_fn=data_collator,
) 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #print(device)


for i, checkpoint in enumerate(model_checkpoints):

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.to(device)

    f1 = open(path  + "/eval_django-10s-" + str(i) + ".txt", "w")

    valid_progress_bar = tqdm(range(len(django_test_intents_dataloader)))

    #reseting sums for next model
    sum_test_sacrebleu = 0
    count =0
    sum_test_chrf = 0
    count2 = 0
    sum_test_accuracy = 0
    

    model.eval()
        
    for batch_id, batch_intent in enumerate(django_test_intents_dataloader):
        valid_progress_bar.update(1)


        #prompt = input("Prompt: ")

        #resseting scores for next sample
        highest_sacrebleu = 0
        highest_sacrebleu_index = 0
        highest_chrf = 0
        highest_chrf_index = 0
        highest_accuracy = 0
        highest_accuracy_index = 0

        batch_intent = {k: v.to(device) for k, v in batch_intent.items()}
        
        sample_outputs = model.generate(
            batch_intent["input_ids"],
            do_sample=True, 
            max_length= len(batch_intent["input_ids"][0])+len(django_tokenized_snippets["test"][batch_id]["input_ids"]),
            top_k=50, 
            top_p=0.95,
            #num_beams = 20,
            num_return_sequences=10,
            temperature = 0.6,
            remove_invalid_values=True,
        )  


        len_real_intent_text = len(tokenizer.decode(batch_intent["input_ids"][0], skip_special_tokens=True))
        len_real_intent_tokens = len(tokenizer.convert_ids_to_tokens(batch_intent["input_ids"][0], skip_special_tokens=True))        
        real_snippet_text = tokenizer.decode(django_tokenized_snippets["test"][batch_id]["input_ids"] , skip_special_tokens=True)
  
        #print("SOLUTION")
        #print(real_snippet_text)
        #print("GENERATED")
        

        for i, sample_output in enumerate(sample_outputs):

            generated_full_text= tokenizer.decode(sample_outputs[i], skip_special_tokens=True)
            generated_snippet_text = generated_full_text[len_real_intent_text:]
            
            #print(generated_snippet_text)
            
            candidates3 = [
                generated_snippet_text 
            ]

            references3 = [
                [real_snippet_text] 
            ]

            candidates4 = sample_outputs[i][len_real_intent_tokens:].cpu().detach().numpy()
            references4 = django_tokenized_snippets["test"][batch_id]["input_ids"].cpu().detach().numpy()

            if len(references4) > 4:
                sacrebleu_eval = datasets.load_metric("sacrebleu") 
                sacrebleu_score = sacrebleu_eval.compute(
                    predictions=candidates3,
                    references=references3,
                )


            if len(references4) == len(candidates4):
                accuracy_eval = datasets.load_metric("accuracy")
                accuracy_score = accuracy_eval.compute(
                    predictions=candidates4,
                    references=references4,
                )
            
            chrf_eval = datasets.load_metric("chrf") 
            chrf_score = chrf_eval.compute(
                predictions=candidates3,
                references=references3,
                word_order=2
            )


            if len(references4) > 4:
                if sacrebleu_score["score"] > highest_sacrebleu:
                    highest_sacrebleu = sacrebleu_score["score"]
                    highest_sacrebleu_index = i
                #print("BLEU " + str(sacrebleu_score["score"]))
                

            if len(references4) == len(candidates4):
                if accuracy_score["accuracy"] > highest_accuracy:
                    highest_accuracy = accuracy_score["accuracy"]
                    highest_accuracy_index = i
                #print("Accuracy " + str(accuracy_score["accuracy"]))


            if chrf_score["score"] > highest_chrf:
                highest_chrf = chrf_score["score"]
                highest_chrf_index = i
            #print("CHRF" + str(chrf_score["score"]))

        if len(references4) > 4:
            sum_test_sacrebleu = sum_test_sacrebleu + highest_sacrebleu
            count = count +1
            
        if len(references4) == len(candidates4):

            sum_test_accuracy = sum_test_accuracy + highest_accuracy
            count2 = count2 + 1

        sum_test_chrf = sum_test_chrf + highest_chrf

    avg_test_sacrebleu = sum_test_sacrebleu/count
    avg_test_accuracy = sum_test_accuracy/count2
    avg_test_chrf = sum_test_chrf/len(django_test_intents_dataloader)


    f1.write("Average SacreBLEU for " + checkpoint + ":\n")
    f1.write(str(avg_test_sacrebleu) + "\n")
    f1.write("Average Accuracy for " + checkpoint + ":\n")
    f1.write(str(avg_test_accuracy) + "\n")
    f1.write("Average CHRF++ for " + checkpoint + ":\n")
    f1.write(str(avg_test_chrf) + "\n\n\n")
    f1.write("count,count2,full_lenght: " + str(count) + " " + str(count2) + " " + str(len(django_test_intents_dataloader)) + "\n")
    model=model.cpu() #guaranteeing the model doesn't stay in GPU so that it doesn't affect the next model load
    f1.close()

