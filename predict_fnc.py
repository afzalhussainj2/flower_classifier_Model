
import torch
from torchvision import models
from torch import nn, optim
from PIL import Image
import numpy as np
import json

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    img = Image.open(image)
    
    # Resize
    img = img.resize((256,256))
    
    # Crop
    img = img.crop((16,16,240,240))
    
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img.transpose(2,0,1)


def load_checkpoint(filepath):
    
    # for gpu disabled device
    checkpoint = torch.load(filepath,map_location=torch.device('cpu'))

    # for collab as gpu is available
    #checkpoint = torch.load(filepath,map_location=torch.device('cuda'))

    model = models.vgg16(weights=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.classifier.parameters(),lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epochs = checkpoint['epochs']
    categories = checkpoint['categories_to_numbers']
    total_inputs = checkpoint['input_size']
    total_outputs = checkpoint['output_size']

    return model,optimizer,epochs,categories,total_inputs,total_outputs

model,optimizer,epochs,categories,total_inputs,total_outputs = load_checkpoint('checkpoint.pth')


def predict(image_path, model, topk=5, category_names=None, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Process image
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.numpy().tolist()[0]
        top_class = top_class.numpy().tolist()[0]

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(i)] for i in top_class]
    else:
        classes = top_class
    
    return top_p, classes
