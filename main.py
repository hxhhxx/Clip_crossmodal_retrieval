import torch
import clip
from torch.utils.data import random_split
from Datasets.Flickr30k import Flickr30k
#from Datasets.MOSCOCO import MOSCOCO
from torch.utils.data import DataLoader
import Evaluation
import parser

def split_dataset(args):
    if args.dataset == "flickr":
        # 加载Flickr数据集
        Dataset= Flickr30k(root='/kaggle/input/flickr30k/images',ann_file="/kaggle/input/flickr30k/captions.txt", transform = preprocess, target_transform =  target_transform )
        dataset_len = len(Dataset)
        #print(len(flickr_Dataset))
        #31783
        train_size, val_size, test_size = dataset_len-2000, 1000, 1000
    # elif args.dataset == "coco":
    #     # 加载COCO数据集
    #     Dataset = 
    #     train_size, val_size, test_size = 4500, 250, 250

    train_dataset, val_dataset, test_dataset = random_split(Dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    target_transform = lambda texts: clip.tokenize(texts[:5])

    k_vals = [1,5,10]

    if args.evaluate:

        print("Start evaluating", flush=True)
        
        _, _, eva_dataset = split_dataset(args)
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


if __name__ == '__main__':
    args = parser.parse_arguments() #read the parameters from parser


    main(args)