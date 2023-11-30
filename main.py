from cgitb import text
from unittest import TextTestResult
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from torch.utils.data import random_split
from Datasets.Flickr30k import Flickr30k
from Datasets.MSCOCO import COCOcaption
from torch.utils.data import DataLoader
import Evaluation
import parser

def split_dataset(args, preprocess, target_transform):
    if args.dataset == "flickr":
        # 加载Flickr数据集

        Dataset= Flickr30k(root='/kaggle/input/flickr30k/images',ann_file="/kaggle/input/flickr30k/captions.txt", transform = preprocess, target_transform =  target_transform )
        dataset_len = len(Dataset)
        #print(len(flickr_Dataset))
        #31783
        train_size, val_size, test_size = dataset_len-2000, 1000, 1000

        train_dataset, val_dataset, test_dataset = random_split(Dataset, [train_size, val_size, test_size])
    elif args.dataset == "coco":
        # 加载COCO数据集
        val_dataset = COCOcaption(root = args.val_root, ann_file = args.val_ann, transform = preprocess, target_transform = target_transform)
        #train_dataset = COCOcaption(args.train_root, args.train_ann, transform = preprocess, target_transform = target_transform)
        #test_dataset = COCOcaption(args.test_root, args.test_ann, transform = preprocess, target_transform = target_transform)

    return  train_dataset, val_dataset ,test_dataset

#https://github.com/openai/CLIP/issues/57 error using Adam optimizer
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def main(args):
    
    k_vals = [1,5,10]
    model, preprocess = clip.load(args.model, device=device)
    target_transform = lambda texts: clip.tokenize(texts[:5])

    if args.evaluate:
        print("Start evaluating", flush=True)

        _,_,eva_dataset = split_dataset(args,preprocess,target_transform)
        eva_Loader = DataLoader(dataset=eva_dataset, batch_size=16, shuffle=False)
        recall_t2i, recall_i2t, mAP_t2i, mAP_i2t = Evaluation.metrics_at_k(model, eva_Loader, k_vals= k_vals, batch_size=16)

        print("Text-to-image Recall@K")
        for k, x in zip(k_vals, recall_t2i):
            print(f" R@{k}: {100*x:.2f}%")

        print("Image-to-text Recall@K")
        for k, x in zip(k_vals, recall_i2t):
            print(f" R@{k}: {100*x:.2f}%")
            
        print("Text-to-image mAP@K")
        for k, x in zip(k_vals, mAP_t2i):
            print(f" mAP@{k}: {100*x:.2f}%")
            
        print("Image-to-text mAP@K")
        for k, x in zip(k_vals, mAP_i2t):
            print(f" mAP@{k}: {100*x:.2f}%")

    train_dataset, val_dataset,_ = split_dataset(args,preprocess,target_transform)
    train_Loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

    for param in model.parameters():
        param.requires_grad = False
    
    model.text_projection.requires_grad = True
    model.visual.proj.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(trainable_params, lr=args.lr, betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    loss = nn.CrossEntropyLoss()
     
    #https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            if p.requires_grad
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 

    for epoch in range(args.num_epoch):
        model.train()

        for images, texts in tqdm(train_Loader):
            optimizer.zero_grad()

            #print(texts.shape) #16*5*77 input text
            # B x 5 x 77 -> 80 x 77 in evaluation
            # image:16*image -> 80*image(3*224*224)

            images = images.repeat_interleave(5, 0)  # Repeat each image 5 times
            texts = torch.flatten(texts, start_dim=0, end_dim=1)

            images = images.to(device)
            texts = texts.to(device)
            
            #same as 
            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            image_loss = loss(logits_per_image, ground_truth)
            text_loss  = loss(logits_per_text, ground_truth)

            total_loss = (image_loss + text_loss) / 2
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        recall_t2i, recall_i2t, mAP_t2i, mAP_i2t = Evaluation.metrics_at_k(model, val_loader, k_vals= k_vals, batch_size=16)

        print("Text-to-image Recall@K")
        for k, x in zip(k_vals, recall_t2i):
            print(f" R@{k}: {100*x:.2f}%")

        print("Image-to-text Recall@K")
        for k, x in zip(k_vals, recall_i2t):
            print(f" R@{k}: {100*x:.2f}%")
            
        print("Text-to-image mAP@K")
        for k, x in zip(k_vals, mAP_t2i):
            print(f" mAP@{k}: {100*x:.2f}%")
            
        print("Image-to-text mAP@K")
        for k, x in zip(k_vals, mAP_i2t):
            print(f" mAP@{k}: {100*x:.2f}%")
    


if __name__ == '__main__':
    args = parser.parse_arguments() #read the parameters from parser

    device = "cuda" if torch.cuda.is_available() else "cpu"


    main(args)