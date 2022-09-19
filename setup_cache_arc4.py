
"""
Developed by Tomas Pimentel Zilhao Pinto e Borges, 201372847
COMP3931 Individual Project
""" 

import os
from pathlib import Path

# setting up ARC4's cache
# run this in every new terminal

hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "/nobackup/sc19tpzp/.cache"), "huggingface"))
)

default_datasets_cache_path = os.path.join(hf_cache_home, "datasets")
HF_DATASETS_CACHE = Path(os.getenv("HF_DATASETS_CACHE", default_datasets_cache_path))

default_datasets_cache_path = os.path.join(hf_cache_home, "datasets")
HF_DATASETS_CACHE = Path(os.getenv("HF_DATASETS_CACHE", default_datasets_cache_path))

default_metrics_cache_path = os.path.join(hf_cache_home, "metrics")
HF_METRICS_CACHE = Path(os.getenv("HF_METRICS_CACHE", default_metrics_cache_path))

default_modules_cache_path = os.path.join(hf_cache_home, "modules")
HF_MODULES_CACHE = Path(os.getenv("HF_MODULES_CACHE", default_modules_cache_path))