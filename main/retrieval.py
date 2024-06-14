import torch
import torch.utils.data as dutils
from typing import List
import align_clip


def encode_dataset(clip, data, device, batch_size = 16):

    with torch.no_grad():
        image_to_text_map = []

        text_to_image_map = []
        if "ms-coco" in data:
            dataset = data['ms-coco']
        elif "flickr" in data:
            dataset = data['flickr']

        dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0

        for images, text in dataloader:
            images = images.to(device)
            text = text.to(device)

            batch_size, captions_per_image, _ = text.shape

            for i in range(batch_size):
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                text_to_image_map += [image_index] * captions_per_image
                image_index += 1
            text = torch.flatten(text, start_dim=0, end_dim=1)
                                        
            image_encodings.append(clip.encode_image(images))
            text_encodings.append(clip.encode_text(text))

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def recall_at_k(clip, dataset: dutils.Dataset, device, k_vals: List[int], batch_size: int):
    print("Encoding all data...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(clip, dataset, device, batch_size=batch_size)
             
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    print("Text-to-image recall...")

    dist_matrix = text_encodings @ image_encodings.T 

    dist_matrix = dist_matrix.cpu()

    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    for k in k_vals:
        topk = inds[:, :k]

        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T

    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    for k in k_vals:
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)
    print("Done.")
    return text_to_image_recall, image_to_text_recall
