# !/usr/bin/env python3
'''
Author: Bijay Gaudel
Date: 09/22/2023
Discription: Simple autoencoder 
'''
import cv2
import torch
import random
import numpy as np 
import torch.nn as nn 
from torch import Tensor 
import torch.optim as optim 
from torchvision import models
from collections import deque


import utils 

class ResNet(models.ResNet):
    def __init__(
        self,
        weights: Tensor = models.resnet.ResNet18_Weights.IMAGENET1K_V1,
        requires_grad: bool = True,
        remove_fc: bool = True,
        show_params: bool = False
    ) -> None:

        super().__init__(
            block=models.resnet.BasicBlock, 
            layers=[2, 2, 2, 2]
            )

        if weights is not None:
            pretrained_model = models.resnet18(weights=weights)
            self.load_state_dict(pretrained_model.state_dict())

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if remove_fc:
            del self.fc

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            x = layer(x)
        return x
    
def encoder(model: str = "resnet"):
    models = {
        "resnet": ResNet,
    }
    
    Model = models[model]
    return Model()

class Decoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        layers = []
        in_channels = config['in_channels']

        for layer_config in config['layers']:
            layers.append(nn.ConvTranspose2d(in_channels, **layer_config))
            layers.append(nn.BatchNorm2d(layer_config['out_channels']))
            in_channels = layer_config['out_channels']

        self.encoder_layers = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(
            in_channels, 
            config['classifier_out_channels'], 
            kernel_size=config['classifier_kernel_size']
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_layers:
            x = self.relu(layer(x))
        x = self.classifier(x)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, model: str='resnet'):
        super().__init__()
        config = utils.load_yaml('./_config.yaml')['decoder_config']
        self.encoder = encoder(model)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding 

def process_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame, dtype=torch.float32) / 255.0
    frame = frame.permute(2,0,1)
    frame = frame.unsqueeze(0)
    return frame

def calculate_similarity(buffer, current_representation, matrice="cosine", num_samples=50):
    sampled_reps = random.sample(buffer, min(num_samples, len(buffer)))

    # Ensure that current_representation is properly flattened
    current_rep_flat = current_representation.view(-1)

    if matrice == "cosine":
        distances = [1 - torch.nn.functional.cosine_similarity(current_rep_flat.unsqueeze(0), past_rep.view(-1).unsqueeze(0), dim=1).item() 
                     for past_rep in sampled_reps]
    elif matrice == "euclidean":
        distances = [torch.norm(current_rep_flat - past_rep.view(-1)).item() for past_rep in sampled_reps]
    else:
        raise ValueError("Your metric should be either 'cosine' or 'euclidean'")
    
    average_distance = sum(distances) / max(len(distances), 1e-8)  # Avoid division by zero
    return average_distance



def get_representation(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation, _ = model(image.to(device))
    print(representation.shape)
    return representation

def train( 
    rehearsel_batch_size = 4, 
    epochs=10, 
    buffer_size = 100,
    learning_rate = 1e-3, 
    fps_interval = 1,
    model_name="resnet",
    video_path = "./5_.MP4",
    similarity_threshold = [0.57000, 0.700],
    ):    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(model=model_name).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    memory_buffer = deque(maxlen=buffer_size)
    
    representation_buffer = deque(maxlen=buffer_size)
    
    def train_step(batch):
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad()
        encoding, outputs = model(batch)
        encoding = encoding.detach()            
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break   
        if frame_counter % int(fps) == 0:
            processed_frame = process_frame(frame)
            representation = get_representation(processed_frame, model)
            
            if frame_counter < 2:
                representation_buffer.append(representation)
                memory_buffer.append(processed_frame)
                frame_filename = f"./sampled_data/frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.png"  
                cv2.imwrite(frame_filename, frame)
            
            similarity = calculate_similarity(representation_buffer, representation)
            print("similarity: ", similarity)
            
            if similarity_threshold[0] < similarity < similarity_threshold[1]:
                frame_filename = f"./sampled_data/frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.png"  
                cv2.imwrite(frame_filename, frame)
                
            if  similarity > 0.55:
                # memory_buffer.append(processed_frame)
                if len(memory_buffer) >= rehearsel_batch_size:
                    rehearsel_frames = random.sample(memory_buffer, rehearsel_batch_size)
                    rehearsel_batch = torch.cat(rehearsel_frames, dim=0)
                    for i in range(epochs):
                        loss = train_step(rehearsel_batch)
                        print(f"iteration: {i}: Rehearsel Training - Loss: {loss}")
                
                if len(memory_buffer) == buffer_size:
                    random_index = random.randint(0, buffer_size-1)
                    del memory_buffer[random_index]
                memory_buffer.append(processed_frame)
                
                if len(representation_buffer) == buffer_size:
                    random_index = random.randint(0, buffer_size-1)
                    del representation_buffer[random_index]
                representation_buffer.append(representation)
        frame_counter += 1

    cap.release()
    
    

if __name__ == '__main__':
    # get_representation()
    train()

