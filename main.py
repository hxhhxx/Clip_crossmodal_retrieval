
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import clip
from torch.utils.data import random_split
from Datasets.Flickr30k import Flickr30k
from torchvision.datasets import CocoCaptions
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
        train_size, val_size, test_size= (dataset_len)*0.94, (dataset_len)*0.03, (dataset_len)*0.03

        train_dataset, val_dataset, test_dataset = random_split(Dataset, [train_size, val_size, test_size])
    
    elif args.dataset == "coco":
        # 加载COCO数据集
        coco = CocoCaptions(root = '/kaggle/input/coco-2017-dataset/coco2017/train2017',
                        annFile = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json',
                        transform=preprocess,
                        target_transform=lambda texts: clip.tokenize(texts[0:5]))        
        train_ratio = 0.95

        train_size = int(train_ratio * len(coco))
        val_size = len(coco) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(coco, [train_size, val_size])

        # coco traning and validation datasets
        coco_test = CocoCaptions(root = '/kaggle/input/coco-2017-dataset/coco2017/val2017',
                                annFile = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json',
                                transform=preprocess,
                                target_transform=lambda texts: clip.tokenize(texts[0:5]))

        test_dataset = DataLoader(coco_test,batch_size=64,shuffle=False,num_workers=4,pin_memory=False)
    
    return  train_dataset, val_dataset ,test_dataset

#https://github.com/openai/CLIP/issues/57 error using Adam optimizer
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


# Added layer

class Adapter(nn.Module):
    def __init__(self, embed_dim, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, embed_dim, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class CustomCLIPModel(nn.Module):
    def __init__(self, clip_model, embed_dim):
        super(CustomCLIPModel, self).__init__()
        self.clip_model = clip_model
        #self.additional_layers = ProjectionHead(embed_dim)
        self.additional_layers = Adapter(embed_dim,4)
        
    def forward(self, image, text):
        # Get features from CLIP
        image_features, text_features = self.clip_model(image, text)               

        # Pass features through additional layers
        x = self.additional_layers(image_features)
        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        #text_features = self.additional_layers(text_features)
        return image_features, text_features


def main(args):
    
    k_vals = [1,5,10]
    model, preprocess = clip.load(args.model, device=device)
    target_transform = lambda texts: clip.tokenize(texts[:5])

    if args.evaluate:
        if args.model_path != "none":
            model.load_state_dict(torch.load(args.model_path))
        model.eval()
        print("Start evaluating", flush=True)

        _,_,eva_dataset = split_dataset(args,preprocess,target_transform)
        eva_Loader = DataLoader(dataset=eva_dataset, batch_size=args.batch_size, shuffle=False)
        Evaluation.metrics_at_k(model, eva_Loader, k_vals= k_vals, batch_size=16)

    train_dataset, val_dataset, test_dataset = split_dataset(args,preprocess,target_transform)
    train_Loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    ####################################
    #change the projection head inside the model

    if args.trainable == "adaptor":

        # Create custom model
        state_dict = model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]
        # print(embed_dim) #512 in b/32 768 in l/14
        new_model = CustomCLIPModel(model, embed_dim).to(device)
        clip.model.convert_weights(new_model)

        for param in model.parameters():
            param.requires_grad = False
        if args.loss == "cross_entropy":
            model.visual.proj.requires_grad = True

        trainable_params = [p for p in new_model.parameters() if p.requires_grad] 

    if args.trainable == "linear_projection":

        for param in model.parameters():
            param.requires_grad = False
        
        model.logit_scale.requires_grad = True 
        model.text_projection.requires_grad = True
        model.visual.proj.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.trainable == "all":

        trainable_params = [p for p in model.parameters() if p.requires_grad]


    ######################################
    #change the Optimizer
         
    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9,0.98), eps=1e-6,weight_decay=1e-3)
    #optimizer = optim.Adam(trainable_params, lr=args.lr, betas=(0.9,0.98), eps=1e-6,weight_decay=1e-3)

    if args.scheduler:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2) 
        #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) 
        
    #https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            if p.requires_grad:
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 

    ##################################################
    #epoch start
    best_val_loss = float('inf')            
    for epoch in range(args.num_epoch):

        total_loss = 0

        if args.trainable == "adaptor":
            new_model.train()
        else :
            model.train()
        print(f"start the {epoch+1}/{args.num_epoch} epoch training")

        for images, texts in tqdm(train_Loader):
            optimizer.zero_grad()

            #texts: batch size x 77
            random_indices = torch.randint(0, 5, (len(images),))
            texts = torch.stack([texts[i, idx] for i, idx in enumerate(random_indices)])

            images = images.to(device)
            texts = texts.to(device)

            #encoding & cosine similarity as logits       
            #

            #encoding & cosine similarity as logits 

            if args.trainable == "adaptor":
                image_encodings, text_encodings = new_model(images, texts)
            else :  
                logits_per_image, logits_per_text = model(images, texts)               
            #     image_encodings = model.encode_image(images)
            #     text_encodings = model.encode_text(texts)

            # image_encodings = image_encodings / image_encodings.norm(dim=1, keepdim=True)
            # text_encodings = text_encodings / text_encodings.norm(dim=1, keepdim=True)

            if args.loss == "cos_embedd":
                cosine_similarity = F.cosine_similarity(image_encodings[:, None, :], text_encodings[None, :, :], dim=2)
                target = torch.eye(cosine_similarity.shape[0],dtype=torch.long, device=device)
                loss = 0.5 * target * (1 - cosine_similarity) + 0.5 * (1 - target) * torch.clamp(cosine_similarity - 0.1, min=0.0)
                loss = torch.mean(loss)

            if args.loss == "cross_entropy":
                # temperature = 0.07
                # logits_per_image = (image_encodings @ text_encodings.t()) / temperature
                # logits_per_text = logits_per_image.T
                targets = torch.arange(len(images),dtype=torch.long, device=device)

                CE_loss = nn.CrossEntropyLoss()
                image_loss = CE_loss(logits_per_image, targets)
                text_loss  = CE_loss(logits_per_text, targets)
                loss = (image_loss + text_loss)/2
                        
            loss.backward()

            total_loss += loss

            #optimizer.step()

            if args.trainable == "adaptor":
                convert_models_to_fp32(new_model)
                optimizer.step()            
                clip.model.convert_weights(new_model)

            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        avg_train_loss = total_loss / len(train_Loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        print(f"Epoch {epoch+1}/{args.num_epoch} - Validation")

        for images, texts in tqdm(val_loader):
            #texts: batch size x 77
            random_indices = torch.randint(0, 5, (len(images),))
            texts = torch.stack([texts[i, idx] for i, idx in enumerate(random_indices)])

            images = images.to(device)
            texts = texts.to(device)
            #encoding & cosine similarity as logits       
            #logits_per_image, logits_per_text = model(images, texts)

            #encoding & cosine similarity as logits 
            if args.trainable == "adaptor":
                image_encodings, text_encodings = new_model(images, texts)
            else :                 
                image_encodings = model.encode_image(images)
                text_encodings = model.encode_text(texts)

            image_encodings = image_encodings / image_encodings.norm(dim=1, keepdim=True)
            text_encodings = text_encodings / text_encodings.norm(dim=1, keepdim=True)

            if args.loss == "cos_embedd":
                cosine_similarity = F.cosine_similarity(image_encodings[:, None, :], text_encodings[None, :, :], dim=2)
                target = torch.eye(cosine_similarity.shape[0],dtype=torch.long, device=device)
                loss = 0.5 * target * (1 - cosine_similarity) + 0.5 * (1 - target) * torch.clamp(cosine_similarity - 0.1, min=0.0)
                loss = torch.sum(loss)

            if args.loss == "cross_entropy":
                temperature = 0.07
                logits_per_image = (image_encodings @ text_encodings.t()) / temperature
                logits_per_text = logits_per_image.T
                targets = torch.arange(len(images),dtype=torch.long, device=device)

                CE_loss = nn.CrossEntropyLoss()
                image_loss = CE_loss(logits_per_image, targets)
                text_loss  = CE_loss(logits_per_text, targets)
                loss = (image_loss + text_loss)/2
            total_val_loss += loss

            if args.scheduler:
                 #lr_scheduler.step(total_val_loss)
                lr_scheduler.step()

        print("start to print the matrix of val for this epoch")
        if args.trainable == "adaptor":
            Evaluation.metrics_at_k(new_model, val_loader, k_vals= k_vals, batch_size=16)
        else:
            Evaluation.metrics_at_k(model, val_loader, k_vals= k_vals, batch_size=16)
             
        ######
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if args.trainable == "adaptor":
                best_model = new_model.state_dict()
            else:
                best_model = model.state_dict()

    
    torch.save(best_model, '/kaggle/working/best_model.pth')
    print("save the best model")

    model, _ = clip.load(args.model, device=device)
    if args.trainable == "adaptor":
        # Create custom model
        state_dict = model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]
        #print(embed_dim) #512 in b/32 
        new_model = CustomCLIPModel(model, embed_dim).to(device)
        clip.model.convert_weights(new_model)

        new_model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))

    else:    
        model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
    print("start to test:")
    if args.trainable == "adaptor":
        Evaluation.metrics_at_k(new_model, test_loader, k_vals= k_vals, batch_size=16)
    else :
        Evaluation.metrics_at_k(model, test_loader, k_vals= k_vals, batch_size=16)



if __name__ == '__main__':
    args = parser.parse_arguments() #read the parameters from parser

    device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args)