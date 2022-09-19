
"""
Developed by Tomas Pimentel Zilhao Pinto e Borges, 201372847
COMP3931 Individual Project
""" 

import json 

fd1= open("/saved_datasets/django/all.anno",mode='r',encoding='utf8', newline='\n')
fd2= open("/saved_datasets/django/all.code",mode='r',encoding='utf8', newline='\n')

fd3= open("/saved_datasets/django_train.jsonl",mode='w',encoding='utf8', newline='\n')
fd4= open("/saved_datasets/django_valid.jsonl",mode='w',encoding='utf8', newline='\n')
fd5= open("/saved_datasets/django_test.jsonl",mode='w',encoding='utf8', newline='\n')
fd6= open("/saved_datasets/django.jsonl",mode='w',encoding='utf8', newline='\n')

# 16000 training
# 1000 validation
# 1805 testing
intents = ["" for x in range(18805)]
snippets = ["" for x in range(18805)]
w = [0 for x in range(543345)]
z = [0 for x in range(543345)]
a = [0 for x in range(543345)]
b = [0 for x in range(543345)]

# use enumerate to show that second line is read as a whole
j = 0
k = 0
c = 0

for i, line in enumerate(fd1):  
    intents[i] = line.strip()
    intents[i]=  intents[i].replace('"',"'" )
    intents[i]=  intents[i].replace('}',"\}" )
    intents[i]=  intents[i].replace('{',"\{" )


for j, line in enumerate(fd2):  
    snippets[j] = line.strip()
    snippets[j] =  snippets[j].replace('"',"'" )
    snippets[j]=  snippets[j].replace('}',"\}" )
    snippets[j]=  snippets[j].replace('{',"\{" )


for k,intent in enumerate(intents):  

    if k < 16000:
        data = {}
        data["intent"] = intents[k]
        data["snippet"] = snippets[k]
        json.dump(data, fd3)
        fd3.write("\n")


for m, intent in enumerate(intents):  

    if m >= 16000 and m < 17000:
        data = {}
        data["intent"] = intents[m]
        data["snippet"] = snippets[m]
        json.dump(data, fd4)
        fd4.write("\n")

for n, intent in enumerate(intents):  

    if  n >= 17000:
        data = {}
        data["intent"] = intents[n]
        data["snippet"] = snippets[n]
        json.dump(data, fd5)
        fd5.write("\n")


fd1.close()
fd2.close()
fd3.close()
fd4.close()
fd5.close()

