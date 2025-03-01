# Mitigate the Gap: Investigating Approaches for Improving Cross-Modal Alignment in CLIP

This is the official implementation of [AlignCLIP](https://openreview.net/forum?id=aPTGvFqile) and provides the source code for pre-training SharedCLIP as well as AlignCLIP.
The implementation is based on the [OpenCLIP](ihttps://github.com/mlfoundations/open_clip).

[**🤗 Checkpoints**](https://huggingface.co/sarahESL/AlignCLIP).


It's recommended to use [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies. Use the following to install the dependencies:

```
conda env create --name envname --file=environments.yml
```
## Pre-training
For pre-training, the `train_{*}.sh` scripts can be used. 

Sample single-process running code for pre-training SharedCLIP on CC12M:

```
python -m main.run --logs="path/to/logs" --save-frequency 2 --report-to wandb --wandb-project-name="sample_project" --train-data="path/to/cc12m" --train-num-samples 10030127 --warmup 10000  --batch-size=512 --lr=1e-3 --wd=0.1 --epochs=30 --workers=2 --model "ViT-B-16" --precision amp --dataset-type webdataset
```

Sample single-process running code for pre-training AlignCLIP on CC12M:

```
python -m main.run --logs="path/to/logs" --save-frequency 2 --report-to wandb --wandb-project-name="sample_project" --train-data="path/to/cc12m" --train-num-samples 10030127 --warmup 10000  --batch-size=512 --lr=1e-3 --wd=0.1 --epochs=30 --workers=2 --model "ViT-B-16" --precision amp --dataset-type webdataset --clip-inModality-loss --clip-loss --alpha=1 --beta=0.5 --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_text --separate_image
```

## Downstream Evaluations
For inference, the `test_{*}.sh` scripts can be used.

Sample single-process inference for zeroshot classification on CIFAR10:
```
python -m main.run --logs="path/to/log/outputs" --pretrained="path/to/pretrained/checkpoint"  --batch-size=4096 --workers=2 --model "ViT-B-16" --cifar10="path/to/cifar"

```

## Citation
If you found this work useful, please cite:
 
```
@inproceedings{
eslami2025mitigate,
title={Mitigate the Gap: Improving Cross-Modal Alignment in {CLIP}},
author={Sedigheh Eslami and Gerard de Melo},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=aPTGvFqile}
}
```

```
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```
