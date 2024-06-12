"""
Adapted from open_clip: https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
"""

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from align_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier
from align_clip import IMAGENET_CLASSNAMES, IMAGENET_A_CLASSNAMES, IMAGENET_R_CLASSNAMES, CIFAR10_CLASSNAMES, CIFAR100_CLASSNAMES, \
        FLOWERS_CLASSNAMES, STANFORD_CLASSNAMES, IMAGENET_O_CLASSNAMES, FOOD_CLASSNAMES,\
        OPENAI_IMAGENET_TEMPLATES, IDENTITY_TEMPLATE
from .precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot classification.')

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        tokenizer = get_tokenizer(args.model)
        if 'cifar10' in data:
            classes = CIFAR10_CLASSNAMES
        elif 'cifar100' in data:
            classes = CIFAR100_CLASSNAMES
        elif 'imagenet-a' in data:
            classes = IMAGENET_A_CLASSNAMES
        elif 'imagenet-o' in data:
            classes = IMAGENET_O_CLASSNAMES
        elif 'imagenet-r' in data:
            classes = IMAGENET_R_CLASSNAMES
        elif 'imagenet-val' in data or 'imagenet-v2' in data or 'imagenet-sketch' in data:
            classes = IMAGENET_CLASSNAMES
        elif 'flowers-102' in data:
            classes = FLOWERS_CLASSNAMES
        elif 'food-101' in data:
            classes = FOOD_CLASSNAMES
        elif 'stanford' in data:
            classes = STANFORD_CLASSNAMES
        else:
            raise ValueError('Unsupported dataset!')
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=classes,
            templates=OPENAI_IMAGENET_TEMPLATES,
            #templates=IDENTITY_TEMPLATE,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    if 'imagenet-sketch' in data:
        top1, top5 = run(model, classifier, data['imagenet-sketch'].dataloader, args)
        results['imagenet-sketch-zeroshot-val-top1'] = top1
        results['imagenet-sketch-zeroshot-val-top5'] = top5
    if 'imagenet-a' in data:
        top1, top5 = run(model, classifier, data['imagenet-a'].dataloader, args)
        results['imagenet-a-zeroshot-val-top1'] = top1
        results['imagenet-a-zeroshot-val-top5'] = top5
    if 'imagenet-o' in data:
        top1, top5 = run(model, classifier, data['imagenet-o'].dataloader, args)
        results['imagenet-o-zeroshot-val-top1'] = top1
        results['imagenet-o-zeroshot-val-top5'] = top5
    if 'imagenet-c' in data:
        top1, top5 = run(model, classifier, data['imagenet-c'].dataloader, args)
        results['imagenet-c-zeroshot-val-top1'] = top1
        results['imagenet-c-zeroshot-val-top5'] = top5
    if 'imagenet-r' in data:
        top1, top5 = run(model, classifier, data['imagenet-r'].dataloader, args)
        results['imagenet-r-zeroshot-val-top1'] = top1
        results['imagenet-r-zeroshot-val-top5'] = top5
    if 'cifar10' in data:
        top1, top5 = run(model, classifier, data['cifar10'].dataloader, args)
        results['cifar10-zeroshot-val-top1'] = top1
        results['cifar10-zeroshot-val-top5'] = top5
    if 'cifar100' in data:
        top1, top5 = run(model, classifier, data['cifar100'].dataloader, args)
        results['cifar100-zeroshot-val-top1'] = top1
        results['cifar100-zeroshot-val-top5'] = top5
    if 'flowers-102' in data:
        top1, top5 = run(model, classifier, data['flowers-102'].dataloader, args)
        results['flowers102-zeroshot-val-top1'] = top1
        results['flowers102-zeroshot-val-top5'] = top5
    if 'food-101' in data:
        top1, top5 = run(model, classifier, data['food-101'].dataloader, args)
        results['food101-zeroshot-val-top1'] = top1
        results['food101-zeroshot-val-top5'] = top5
    if 'stanford' in data:
        top1, top5 = run(model, classifier, data['stanford'].dataloader, args)
        results['stanford-zeroshot-val-top1'] = top1
        results['stanford-zeroshot-val-top5'] = top5


    logging.info('Finished zero-shot.')
    logging.info(f'The results are: {results}.')

    return results
