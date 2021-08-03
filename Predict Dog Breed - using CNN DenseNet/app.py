import pickle
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from flask import Flask, request
import requests

model = None
app = Flask(__name__)

def load_model():
    global model
    # model variable refers to the global variable
    with open('densenet_model.pkl', 'rb') as f:
        model = pickle.load(f)

# Preprocess function
def preprocess_image(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch


@app.route('/predict', methods=['POST','GET'])
def get_prediction():
    if request.method == 'POST' or request.method == 'GET':
        data = preprocess_image(Image.open(request.files['file']))
        with torch.no_grad():
            prediction = model(data)  # runs globally loaded model on the data

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)

        with open("./imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top_prob, top_catid = torch.topk(probabilities, 1)
    return categories[top_catid[0]]

@app.route('/')
def home_endpoint():
    return 'Predict Dogs!'

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)