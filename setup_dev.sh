
"""
Developed by Tomas Pimentel Zilhao Pinto e Borges, 201372847
COMP3931 Individual Project
""" 

#assumption_ conda is available

conda create --name tzb_fyp python=3.6
conda activate tzb_fyp
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
conda install -c conda-forge transformers
conda install -c huggingface -c conda-forge datasets
conda install importlib-metadata
conda install -c fastai accelerate
conda install -c conda-forge gputil
conda install -c conda-forge git-lfs