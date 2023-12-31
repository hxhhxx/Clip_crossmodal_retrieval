
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from pytorch_metric_learning import losses
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

# projection layer
class proj_layer(nn.Module):
    def __init__(self, clips_model):
        super().__init__()
        self.text_proj = clips_model.text_projection
        self.image_proj = clips_model.visual.proj

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = image @ self.image_proj
        text_features = text @ self.text_proj
        return image_features, text_features

class new_projection(nn.Module):
    def __init__(
        self,
        width,
        output_dim,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(width, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
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
        model.eval()
        print("Start evaluating", flush=True)

        _,_,eva_dataset = split_dataset(args,preprocess,target_transform)
        eva_Loader = DataLoader(dataset=eva_dataset, batch_size=args.batch_size, shuffle=False)
        Evaluation.metrics_at_k(model, eva_Loader, k_vals= k_vals, batch_size=args.batch_size)

    train_dataset, val_dataset,_ = split_dataset(args,preprocess,target_transform)
    train_Loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    ####################################
    #change the projection inside the model

    if args.trainable == "new_projection":
    
        image_projection =  new_projection(width=768, output_dim=77, dropout=0.1)
        text_projection = new_projection(width=77, output_dim=77, dropout=0.1)
        
        model.text_projection = None
        model.visual.proj = None

        for param in model.parameters():
            param.requires_grad = False

        trainable_params = list(image_projection.parameters()) + list(text_projection.parameters())

    if args.trainable == "linear_projection":

        for param in model.parameters():
            param.requires_grad = False
            
        model.text_projection.requires_grad = True
        model.visual.proj.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.trainable == "all":

        trainable_params = [p for p in model.parameters() if p.requires_grad]

    ######################################
    #Optimizer
         
    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9,0.98), eps=1e-6,weight_decay=1e-3)
    #optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=1, factor=0.8
    # )
       
    ######################################
    #Loss function 

    CE_loss = nn.CrossEntropyLoss()
    contrastive_loss = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1)

    def CE_loss_logsoftmax(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        

    #https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            if p.requires_grad:
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 

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
            #logits_per_image, logits_per_text = model(images, texts)
            
            #encoding & cosine similarity as logits       
            image_encodings = model.encode_image(images)
            text_encodings = model.encode_text(texts)

            if args.trainable == "new_projection":
                image_encodings = image_projection(image_encodings)
                text_encodings = text_projection(text_encodings)
       
            # Normalise 
            image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
            text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

            temperature = 0.07
            logits_per_image = (image_encodings @ text_encodings.T)/ temperature
            logits_per_text = logits_per_image.T
            
            ##############################################
            #Change the loss function
            if args.loss == "logsoftmax" :
                images_similarity = image_encodings @ image_encodings.T
                texts_similarity = text_encodings @ text_encodings.T
                targets = F.softmax(
                    (images_similarity + texts_similarity) / 2 , dim=-1
                )
                texts_loss = CE_loss_logsoftmax(logits_per_text, targets, reduction='none')
                images_loss = CE_loss_logsoftmax(logits_per_image, targets.T, reduction='none')
                loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
                #loss.mean()

            if args.loss == "cross_entropy" :
                targets = torch.arange(len(images),dtype=torch.long, device=device)

                image_loss = CE_loss(logits_per_image, targets)
                text_loss  = CE_loss(logits_per_text, targets)
                loss = (image_loss + text_loss)/2

            if args.loss == "contrastive" :
                
                targets = torch.arange(len(images))

                image_loss = contrastive_loss(logits_per_image , targets)
                text_loss = contrastive_loss(logits_per_text , targets)
                loss = (image_loss + text_loss)/2
            
                # loss = contrastive_loss(logits_per_image, logits_per_text)
            
            loss.backward()

            total_loss += loss

            if device == "cpu":
                optimizer.step()
                
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        avg_loss = total_loss / len(train_Loader)
        print(f"Epoch {epoch+1}/{args.num_epoch} has done, Average Loss: {avg_loss}")

        model.eval()
        print("start to evaluate")
        Evaluation.metrics_at_k(model, val_loader, k_vals= k_vals, batch_size=16)


if __name__ == '__main__':
    args = parser.parse_arguments() #read the parameters from parser

    device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args)