
"""
Developed by Tomas Pimentel Zilhao Pinto e Borges, 201372847
COMP3931 Individual Project
""" 

# ARC4 batch job submssion file

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=48:00:00

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=18G

#ask for gpu
#$ -l coproc_v100=1
# Send emails when job starts and ends
#$ -m be

# Now run the job
python eval_django.py > output.$JOB_ID.txt