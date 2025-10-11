from data_loaders.reward_model import load_hh_rlhf_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import torch.nn as nn
import torch.nn.functional as F
import torch

def train_reward_model():
    pass