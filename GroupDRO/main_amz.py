import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from torchvision.models import resnet50, ResNet50_Weights, densenet121
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast
from torchvision import transforms

import os
from copy import deepcopy
from tqdm import tqdm
import argparse
import random
from collections import defaultdict

from loss_function import * #SONEX, SONX, OOA
from utils import *

def set_seed(seed=2024):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_bert_transform():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform

# Training loop
def train_one_epoch(args, epoch, model, loss_fn, optimizer, scheduler, dataloader, grouper, eval_loader=None, k=None):
    global aux_ce_loss
    model.train()
    warmup_steps = args.warmup_epoch * len(dataloader)
    eval_freq = 1000 
    log_train_loss, log_train_acc = [], []

    print(f"lr for curr epoch: {optimizer.param_groups[0]['lr']}")
    for curr_iter, batch in enumerate(dataloader, start=1):
        inputs, targets, metadata = batch
        inputs, targets = inputs.cuda(), targets.cuda()
        group_idx = grouper.metadata_to_group(metadata)
        input_ids = inputs[:, :, 0]
        attention_mask = inputs[:, :, 1]

        # Compute the warmup learning rate
        if epoch < args.warmup_epoch:
            current_step = epoch * len(dataloader) + curr_iter
            lr = args.lr * (current_step / warmup_steps)  # Linear warmup

            # Update optimizer's learning rate
            for group in optimizer.param_groups:
                group['lr'] = lr
        if curr_iter % eval_freq == 0:
            print(f"Evaluate at epoch {epoch}, Iteration {curr_iter}:")
            train_loss, train_acc = evaluate(model, eval_loader, grouper, k)
            log_train_loss.append(train_loss)
            log_train_acc.append(train_acc)
            model.train()
        
        if args.algorithm in ['SONX', 'SONEX']:
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)[0]
            if args.algorithm == 'SONEX':
                _, aux_ce_loss = loss_fn(epoch, outputs, targets, group_idx)
            else:
                aux_ce_loss = loss_fn(epoch, outputs, targets, group_idx)
            try:        
                optimizer.step()  
            except:
                pass

            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)[0]
            loss = loss_fn(epoch, outputs, targets, group_idx, aux_ce_loss)
            loss.backward()
        elif args.algorithm in ['OOA']:
            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)[0]
            loss = loss_fn(outputs, targets, group_idx)
            loss.backward()
            optimizer.step()

    if epoch >= args.warmup_epoch:
        scheduler.step()

    return log_train_loss, log_train_acc

def evaluate(model, dataloader, grouper, k):
    model.eval()
    total_correct = 0
    total_samples = 0

    group_losses, group_corrects, group_accuracies, group_num = defaultdict(float), defaultdict(float), {}, defaultdict(int)

    with torch.no_grad():
        # for batch in tqdm(dataloader):
        for batch in dataloader:
            inputs, targets, metadata = batch
            inputs, targets = inputs.cuda(), targets.cuda()
            group_idx = grouper.metadata_to_group(metadata)  
            input_ids = inputs[:, :, 0]
            attention_mask = inputs[:, :, 1]

            # Forward pass
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)[0]
            predictions = torch.argmax(outputs, dim=1)  # Get class predictions

            # Compute losses and accuracy
            ce_loss = nn.CrossEntropyLoss(reduction="none")(outputs, targets)

            # Per-group metrics
            groups_in_batch = torch.unique(group_idx).cpu().numpy()

            for group in groups_in_batch:
                group_mask = (group_idx == group)
                group_num[group] += group_mask.sum().item()
                group_losses[group] += ce_loss[group_mask].sum().item() 
                group_corrects[group] += (predictions[group_mask] == targets[group_mask]).sum().item()

    for group in group_losses.keys():
        group_accuracies[group] = group_corrects[group] / group_num[group]
        group_losses[group] /= group_num[group]
    # Sort groups by loss and accuracy
    group_losses = sorted(group_losses.items(), key=lambda item: item[1], reverse=True)
    group_accuracies = sorted(group_accuracies.items(), key=lambda item: item[1])

    # Compute overall metrics
    
    mean_topk_acc = sum(value for _, value in group_accuracies[:k]) / k
    mean_topk_loss = sum(value for _, value in group_losses[:k]) / k

    print(f"TopK Loss: {mean_topk_loss:.4f}, TopK Accuracy: {mean_topk_acc:.4f}")
    print(f"Top-{k} Worst Groups by Loss:")
    print("Group:")
    for group_loss in group_losses[:k:10]:
        print(f" {group_loss[0]}", end=",")
    print("\nLoss:")
    for group_loss in group_losses[:k:10]:
        print(f" {group_loss[1]:.4f}", end=",")

    print("\nGroup:")
    print(f"Top-{k} Worst Groups by Accuracy:")
    for group_acc in group_accuracies[:k:10]:
        print(f" {group_acc[0]}", end=",")
    print("\nAccuracy:")
    for group_acc in group_accuracies[:k:10]:
        print(f" {group_acc[1]:.4f}", end=",")
    print("\n")

    return mean_topk_loss, mean_topk_acc

def main(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Amazon-WILDS dataset
    dataset = get_dataset("amazon", download=True)
    # Define transforms
    train_transform = get_bert_transform()
    eval_transform = train_transform

    # Train, validation, and test splits
    train_data = dataset.get_subset("train", transform=train_transform)
    val_data = dataset.get_subset("val", transform=eval_transform)
    test_data = dataset.get_subset("test", transform=eval_transform)

    grouper = CombinatorialGrouper(dataset, groupby_fields=["user"])
    n_groups, n_groups_val, n_groups_test = 1252, 1334, 1334 # number of groups in train set is actually 150
    batch_size, n_groups_per_batch, val_bsz = 32, 8, 64
    train_loader = get_train_loader("group", train_data, grouper=grouper, batch_size=batch_size, \
                                    n_groups_per_batch=n_groups_per_batch, num_workers=4)
    train_loader_eval = get_eval_loader("standard", train_data, batch_size=val_bsz, num_workers=4)
    val_loader = get_eval_loader("standard", val_data, batch_size=val_bsz, num_workers=2)
    test_loader = get_eval_loader("standard", test_data, batch_size=val_bsz, num_workers=2)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=dataset.n_classes)
    model = model.to(device)

    if args.algorithm == 'SONEX':
        loss_fn = SONEX(n_groups=n_groups, n_groups_per_batch = n_groups_per_batch, \
                       alpha=args.alpha, gamma=args.gamma, theta=args.theta, lamda=args.lamda, lr_c=args.lr_c)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(1-args.beta, 0.999))
    elif args.algorithm == 'SONX':
        loss_fn = SONX(n_groups=n_groups, n_groups_per_batch = n_groups_per_batch, \
                       alpha=args.alpha, gamma=args.gamma, theta=args.theta)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.algorithm == 'OOA': #OOA refer to PrimalDual in paper
        loss_fn = OOA(n_groups=n_groups, n_groups_per_batch = n_groups_per_batch, \
                       alpha=args.alpha, gamma=args.gamma)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=args.epoch)

    num_epochs = args.epoch
    k = round(args.alpha*n_groups_val)
    log_train_loss, log_train_acc = [], []
    log_val_loss, log_val_acc = [], []
    print("Training Started")
    best_val_acc, best_epoch = 0,0
    for epoch in range(num_epochs):
        epoch_log = train_one_epoch(args, epoch, model, loss_fn, optimizer, scheduler, train_loader, grouper, train_loader_eval, k)
        print(f"Evaluation after Epoch {epoch}:")
        print(f"Evaluation for val:")
        val_loss, val_acc = evaluate(model, val_loader, grouper, k)
        log_train_loss.extend(epoch_log[0])
        log_train_acc.extend(epoch_log[1])
        log_val_loss.append(val_loss)
        log_val_acc.append(val_acc)
        
        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_acc
            model_best = deepcopy(model.state_dict())
            print("Best model updated with val acc:", best_val_acc)
        
        log_train_acc, log_train_loss = [float(f"{x:.4f}") for x in log_train_acc], [float(f"{x:.4f}") for x in log_train_loss]
        log_val_acc, log_val_loss = [float(f"{x:.4f}") for x in log_val_acc], [float(f"{x:.4f}") for x in log_val_loss]
        print(f'current trainlog for epoch{epoch}:\nacc: {log_train_acc}\n loss: {log_train_loss}')
    

    # Final test evaluation
    print("Final Test Evaluation:")
    model.load_state_dict(model_best)
    _, test_acc = evaluate(model, test_loader, grouper, k)
    print(f'train:\nacc: {log_train_acc}\n loss: {log_train_loss}')
    print(f'val:\nacc: {log_val_acc}\n loss: {log_val_loss}')
    print(f'Final test acc: {test_acc} at epoch {best_epoch}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--algorithm', '--alg', type=str, default='OSSO', help='Algorithm name')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--epoch', type=int, default=12, help='number of epochs')
    parser.add_argument('--warmup_epoch', '--wup', type=int, default=1, help='number of epoch to do warmup')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')  
    parser.add_argument('--lr_c', type=float, default=0.1, help='learning rate for c')    
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.150, help='alpha: CVaR persentile')
    parser.add_argument('--beta', type=float, default=0.8, help='mmt/STORM coef')
    parser.add_argument('--gamma', type=float, default=0.2, help='dual lr/inner est step size')
    parser.add_argument('--theta', type=float, default=0.1, help='inner est extrapolation coef')
    parser.add_argument('--lamda', type=float, default=0.01, help='smoothing coef')
    parser.add_argument('--eps', type=float, default=1e-2, help='eps for AdamW')
    
    parser.add_argument('--gpu', type=str, default='1', help='GPU number')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)
    global aux_ce_loss 
    aux_ce_loss= -1.0
    main(args)