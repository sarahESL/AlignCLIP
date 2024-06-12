import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    
from .scheduler import cosine_lr


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
def get_linear_probe_metrics(model, data, args):

    metric = "accuracy"
    device = args.device

    if 'cifar10' in data:
        test_dataloader = data['cifar10'].dataloader
        train_dataloader = data['cifar10-train'].dataloader
        output_dim = 10
    elif 'cifar100' in data:
        test_dataloader = data['cifar100'].dataloader
        train_dataloader = data['cifar100-train'].dataloader
        output_dim = 100
    elif 'imagenet-val' in data:
        test_dataloader = data['imagenet-val'].dataloader
        train_dataloader = data['imagenet-train'].dataloader
        output_dim = 1000
    elif 'flowers-102' in data:
        test_dataloader = data['flowers-102'].dataloader
        train_dataloader = data['flowers-102-train'].dataloader
        output_dim = 102
    elif 'food-101' in data:
        test_dataloader = data['food-101'].dataloader
        train_dataloader = data['food-101-train'].dataloader
        output_dim = 101
    elif 'stanford' in data:
        test_dataloader = data['stanford'].dataloader
        train_dataloader = data['stanford-train'].dataloader
        output_dim = 196
    else:
        raise ValueError('Unsupported dataset!')

    #input_dim = 512  ## Vor ViT-B 
    input_dim = 768
    
    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(args.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_lr(optimizer, 0.005, 0, len(train_dataloader) * args.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    pbar = tqdm(range(args.linear_probe_num_epochs))
    for epoch in pbar:
        cbar = tqdm(train_dataloader, leave = False)
        for index, (image, label) in enumerate(cbar):
            image = image.to(device)
            label = label.to(device)
            step = len(train_dataloader) * epoch + index
            scheduler(step)
            output = model(image=image)
            image_features = output['image_features'] if isinstance(output, dict) else output[0]
            image, label = image_features.to(args.device), label.to(args.device)
            logit = classifier(image)
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    classifier.eval()
    
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            test_size = 0
            for image, label in tqdm(test_dataloader):
                test_size += image.shape[0]
                image, label = image.to(args.device), label.to(args.device)
                output = model(image=image)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = classifier(image_features)
                prediction = torch.argmax(logits, dim = 1)
                correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_size}
        else:
            correct = torch.zeros(output_dim).to(args.device)
            total = torch.zeros(output_dim).to(args.device)
            for image, label in tqdm(test_dataloader):
                image, label = image.to(args.device), label.to(args.device)
                output = model(image=image)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = classifier(image_features)
                predictions = torch.argmax(logits, dim = 1)
                
                temp = torch.zeros(output_dim, len(label)).to(args.device)
                temp[label, torch.arange(len(label))] = (predictions == label).float()
                correct += temp.sum(1)
                temp[label, torch.arange(len(label))] = 1                
                total += temp.sum(1)

            results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
        
    logging.info("Finished linear probe testing")
    return results
