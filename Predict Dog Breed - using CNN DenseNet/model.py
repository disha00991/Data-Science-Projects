import torch
import pickle

def save_model():
    model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
    model.eval()

    with open('densenet_model.pkl', 'wb') as f:
        pickle.dump(model, f)
