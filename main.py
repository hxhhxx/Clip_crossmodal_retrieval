
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
#from pytorch_metric_learning import losses
import clip
from torch.utils.data import random_split
from Datasets.Flickr30k import Flickr30k
#from Datasets.MSCOCO import COCOcaption
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

# new projection layer
class proj_layer(nn.Module):
    def __init__(self, clips_model):
        super().__init__()
        self.text_proj = clips_model.text_projection
        self.image_proj = clips_model.visual.proj
    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = image @ self.image_proj
        text_features = text @ self.text_proj
        return image_features, text_features

def contrastive_loss(logits_per_image, logits_per_text, margin=1.0):
    #这个结果很差：猜测可能是前五个的similarities结果并不好并不是positive，similarities are closed 
    # distance_per_image = margin - logits_per_image
    # distance_per_text = margin - logits_per_text

    # distance_per_image,_ = torch.sort(distance_per_image, dim=1, descending=False)
    # distance_per_text,_ = torch.sort(distance_per_text, dim=1, descending=False)
    
    # #print(distance_per_image)
    # # [0.6895, 0.6982, 0.7227,  ..., 0.9053, 0.9058, 0.9399] similarities are closed 

    # # loss of the positive pairs
    # positive_loss_image = distance_per_image[:, :5].mean()
    # positive_loss_text = distance_per_text[:, :1].mean()

    # # loss of the negative pairs
    # negative_loss_image = F.relu(margin - logits_per_image[:, 5:]).mean()
    # negative_loss_text = F.relu(margin - logits_per_text[:, 1:]).mean()


    text_i_matrix = torch.eye(len(logits_per_image)).repeat_interleave(5,dim=0).to(device)
    image_i_matrix = torch.transpose(text_i_matrix, 0, 1).to(device)

    distance_per_image = margin - logits_per_image
    distance_per_text = margin - logits_per_text

    positive_loss_image = (distance_per_image * image_i_matrix).mean()
    positive_loss_text = (distance_per_text * text_i_matrix).mean()

    negative_loss_image = F.relu(distance_per_image * (1-image_i_matrix)).mean()
    negative_loss_text = F.relu(distance_per_text * (1-text_i_matrix)).mean()

    total_loss= positive_loss_image + negative_loss_image + positive_loss_text + negative_loss_text

    return total_loss/2

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

    for param in model.parameters():
        param.requires_grad = False
    
    model.text_projection.requires_grad = True
    model.visual.proj.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    #proj = proj_layer(model)
    optimizer = optim.Adam(trainable_params, lr=args.lr, betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    CE_loss = nn.CrossEntropyLoss()

    #https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            if p.requires_grad:
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 

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
            
            # Normalise 
            image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
            text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

            temperature = 0.07
            logits_per_image = (image_encodings @ text_encodings.T)/ temperature
            logits_per_text = logits_per_image.T
            
            if args.loss == "cross_entropy" :
                targets = torch.arange(len(images),dtype=torch.long, device=device)

                image_loss = CE_loss(logits_per_image, targets)
                text_loss  = CE_loss(logits_per_text, targets)
                loss = (image_loss + text_loss)/2

            if args.loss == "contrastive" :
                
                #使用pytorch_metric_learning库结果和我自己写的第一个一样很差，猜测可能是相似值过于靠近
                #targets_images = torch.arange(len(images))
                #targets_texts = targets_images.repeat_interleave(5)

                #image_loss = contrastive_loss(logits_per_image , targets_images)
                #text_loss = contrastive_loss(logits_per_text , targets_texts)
                #loss = (image_loss + text_loss)/2
                
                #contrastive loss from define
                loss = contrastive_loss(logits_per_image, logits_per_text)
            
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