# configuration
import os

path = os.path.realpath(__file__)
dir = os.path.dirname(path)

input = dir.replace("src","input")
TRAINING_FILE = os.path.join(input,"emotion_dataset.csv")

MODEL_OUTPUT = dir.replace("src","models")
print(MODEL_OUTPUT)