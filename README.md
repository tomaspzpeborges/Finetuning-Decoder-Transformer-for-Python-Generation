# Finetuning Decoder Transformer for Python Generation 


This project intends to finetune a pre-trained decoder-only transformer model to the downstream task of python generation. 
We successfully finetune a decoder-only transformer - CodeParrot-CoNaLa-16-Django-11-100 - into generating one-line python snippets from pseudocode-like descriptions, with no guarantees of completeness. We report a BLEU score of 41.26 on the Django test dataset and it generates higher quality snippets than its baseline and other models of comparable size and pretraining.

See report for further details. 
