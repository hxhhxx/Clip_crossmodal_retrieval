from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import clip
from torch.utils.data import random_split
from Datasets.Flickr30k import Flickr30k
#from Datasets.MSCOCO import COCOcaption
from torch.utils.data import DataLoader
import Evaluation
import parser
import itertools

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

def main(args):
    
    k_vals = [1,5,10]
    model, preprocess = clip.load(args.model, device=device)
    target_transform = lambda texts: clip.tokenize(texts[:5])

    train_dataset, val_dataset,_ = split_dataset(args,preprocess,target_transform)
    train_Loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    ####################################
    #change the projection head inside the model

    if args.trainable == "linear_projection":

        for param in model.parameters():
            param.requires_grad = False
            
        model.text_projection.requires_grad = True
        model.visual.proj.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
  ######################################
    #change the Optimizer
         
    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9,0.98), eps=1e-6,weight_decay=1e-3)
   ######################################
    #Loss function define (change inside the epoch)

    CE_loss = nn.CrossEntropyLoss()
        
    ##################################################
    #epoch start
                
    for epoch in range(args.num_epoch):
        total_loss = 0

        model.train()
        print("start to train")

        for images, texts in tqdm(train_Loader):
            optimizer.zero_grad()

            #texts: batch size x 77
            random_indices = torch.randint(0, 5, (len(images),))
            texts = torch.stack([texts[i, idx] for i, idx in enumerate(random_indices)])

            images = images.to(device)
            texts = texts.to(device)

            #encoding & cosine similarity as logits       
            logits_per_image, logits_per_text = model(images, texts)

            if args.loss == "cross_entropy" :
                targets = torch.arange(len(images),dtype=torch.long, device=device)

                image_loss = CE_loss(logits_per_image, targets)
                text_loss  = CE_loss(logits_per_text, targets)
                loss = (image_loss + text_loss)/2
            
            loss.backward()

            total_loss += loss.item()

            optimizer.step()

        model.eval()
        print("start to evaluate")
        Evaluation.metrics_at_k(model, val_loader, k_vals= k_vals, batch_size=16)

        avg_train_loss = total_loss / len(train_Loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

if __name__ == '__main__':
    args = parser.parse_arguments() #read the parameters from parser

    device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args)