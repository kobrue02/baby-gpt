from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from contextlib import nullcontext

from inference.run_model import load_model, generate
