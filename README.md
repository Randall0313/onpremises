This repository contains code for predicting flight delays. The code preprocesses flight data, trains a model, and evaluates performance using metrics like accuracy, precision, recall, and ROC AUC.

## Requirements
To install the required Python packages:
import os
from pathlib2 import Path
from zipfile import ZipFile
import time

import pandas as pd
import numpy as np
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
