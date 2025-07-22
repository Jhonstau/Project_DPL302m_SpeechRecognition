import numpy as np
import pandas as pd
import os
import pickle
import random
from copy import deepcopy
import librosa
import librosa.util
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def load(path):
    model = torch.load(path, weights_only=False, map_location='cpu')
    return model.eval()

def predict(X_batch, model):
    inputs = torch.tensor(X_batch, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    emotion_map = {
        1: "neutral", 2: "calm", 3: "happy",
        4: "sad", 5: "angry", 6: "fear",
        7: "disgust", 8: "surprised"
    }
    result = []
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = [emotion_map[p + 1] for p in preds]
        result.append(labels)
    return result
    