import os
import argparse

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

########
# TBU
########

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default='')
    parser.add_argument("--retrieved_chunks", type=list, default=[])
    parser.add_argument("--generated_answer", type=str, default='')
    args = parser.parse_args()

