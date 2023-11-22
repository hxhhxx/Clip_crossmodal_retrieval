import torch
import numpy as np
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

# Encodes all text and images in a dataset
def encode_dataset(clip,  eva_Loader, batch_size = 16):

    with torch.no_grad():
        #  gives the corresponding text indices for the ith image and text
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []
        text_to_image_map = []

        #dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0

        for images, text, image_id in eva_Loader:
            images = images.to(device)
            text = text.to(device)

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = text.shape

            # Update text_to_image_map and image_to_text_map for this batch(16 images.....)
            for i in range(batch_size):
                
                #index(map)
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            
            image_encodings.append(clip.encode_image(images))
            text_encodings.append(clip.encode_text(text))

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        # Normalise 
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def metrics_at_k(clip, eva_Loader, k_vals: List[int], batch_size: int):
    print("Encoding all data...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(clip, eva_Loader, batch_size=batch_size)
 
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    
    #点乘（cos similarity）
    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    
    #print("Text-to-image recall...")
    
    text_to_image_recall = []

    for k in k_vals:
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    #print("Text-to-image mAP...")
    
    mAP_t2i=[]
    
    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]
        precision_calculator = 0
        #num_correct_calculator = 0

        for i in range(k):
            #print(i)
            the_ith_retrieval = topk[:, i] #变成了一维

            # Correct iff one of the 第i行top_k values equals the correct image (as given by text_to_image_map)
            correct = torch.eq(the_ith_retrieval, text_to_image_map).cpu()
            num_correct = torch.sum(correct).item()
            #print(num_correct)
            #num_correct_calculator += num_correct
            
            precision = num_correct / (i+1)
            precision_calculator += precision
        AP_sum = precision_calculator / 1
        mAP_t2i.append(AP_sum / num_text)


    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    
    #print("Image-to-text recall...")

    image_to_text_recall = []

    for k in k_vals:
        
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#
        

    #print("Image-to-text mAP...")

    mAP_i2t=[]
    for k in k_vals:
        
        topk = inds[:, :k]
        precision_calculator = 0
        #num_correct_calculator = 0

        AP = []

        for im in range(num_im):
            #一张张来
            topk_indices = topk[im]  # 带着他的k个描述
            relevant_indices = image_to_text_map[im]  # 五个relevant captions for this image

            num_relevant_items_found = 0
            precision_at_i = 0
            sum_precisions = 0
            num_precisions = 0

            for rank, prediction in enumerate(topk_indices):
                if prediction in relevant_indices:
                    num_relevant_items_found += 1
                    precision_at_i = num_relevant_items_found / (rank + 1)
                    sum_precisions += precision_at_i
                    num_precisions += 1

            average_precision = sum_precisions / 5 #num_precisions if num_precisions else 0
            #or 5?? but if it's 5 mAP1 will be very small
            AP.append(average_precision)

        mAP_i2t.append(sum(AP) / len(AP))

    print("Done.")
    return text_to_image_recall, image_to_text_recall, mAP_t2i, mAP_i2t




