import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False, semantic_features=None):

        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        clip_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        )/2
        return {"contrastive_loss": clip_loss} if output_dict else clip_loss


class ClipInModalityLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            alpha=1.0,
            beta=0.5,
            n_epoch=30,
            nl_semantic_supervision=False,
            separate_text=True,
            separate_image=False
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        
        self.alpha = alpha
        self.beta = beta
        
        self.nl_semantic_supervision = nl_semantic_supervision
        self.separate_text = separate_text
        self.separate_image = separate_image

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = image_features @ all_image_features.T
                logits_per_text = text_features @ all_text_features.T

                logits_image_text = image_features @ text_features.T
                size = logits_per_image.shape[0]

                logscale_logits_image_text = logit_scale * image_features @ all_text_features.T
                logscale_logits_text_image = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = all_image_features @ all_image_features.T
                logits_per_text = all_text_features @ all_text_features.T

                logits_image_text = all_image_features @ all_text_features.T
                size = logits_per_image.shape[0]
                
                logscale_logits_image_text = logit_scale * all_image_features @ all_text_features.T
                logscale_logits_text_image = logscale_logits_image_text.T 
        else:
            logits_per_image = image_features @ image_features.T
            logits_per_text = text_features @ text_features.T
            
            logits_image_text = image_features @ text_features.T
            size = logits_per_image.shape[0]
            
            logscale_logits_image_text = logit_scale * image_features @ text_features.T
            logscale_logits_text_image = logit_scale * text_features @ image_features.T
        
        
        return logits_per_image, logits_per_text, logits_image_text, logscale_logits_image_text, logscale_logits_text_image

    def forward(self, image_features, text_features, logit_scale, output_dict=False, semantic_features=None):
        device = image_features.device
        logits_per_image, logits_per_text, logits_image_text, logscale_logits_image_text, logscale_logits_text_image = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        if self.nl_semantic_supervision:

            semantic_features = semantic_features / semantic_features.norm(dim=-1, keepdim=True)
            semantic_sim = semantic_features @ semantic_features.T
            semantic_sim = 1 - semantic_sim

            device = logits_per_image.get_device()

            size = logits_per_image.shape[0]

            logits_paired_text_image = torch.mul(logits_image_text, torch.eye(size).to(device))

            if self.separate_text:
                logits_per_text = torch.mul(logits_per_text, semantic_sim)
                logits_per_text = logits_per_text + logits_paired_text_image
                logscale_logits_per_text = logit_scale * logits_per_text

            if self.separate_image:
                logits_per_image = torch.mul(logits_per_image, semantic_sim)
                logits_per_image = logits_per_image + logits_paired_text_image
                logscale_logits_per_image = logit_scale * logits_per_image

            if self.separate_text and self.separate_image:
                inModality_loss = self.beta*(F.cross_entropy(logscale_logits_per_text, labels) +
                        F.cross_entropy(logscale_logits_per_image, labels))
            elif self.separate_text:
                inModality_loss = self.beta*(F.cross_entropy(logscale_logits_per_text, labels))
            elif self.separate_image:
                inModality_loss = self.beta*(F.cross_entropy(logscale_logits_per_image, labels))


        else:
            logscale_logits_image = logit_scale * logits_per_image
            logscale_logits_text = logit_scale * logits_per_text

            inModality_loss = self.beta*(
                F.cross_entropy(logscale_logits_image, labels) +
                F.cross_entropy(logscale_logits_text, labels)
                )
        
        clip_loss = self.alpha*((
            F.cross_entropy(logscale_logits_image_text, labels) +
            F.cross_entropy(logscale_logits_text_image, labels)))


        total_loss = inModality_loss + clip_loss
        return {"total_loss": total_loss, "clip_loss": clip_loss, "inModality_loss": inModality_loss} if output_dict else total_loss
