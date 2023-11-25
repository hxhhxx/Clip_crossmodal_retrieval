# Clip_crossmodal_retrieval

The main objective of this project is to set up a crossmodal retrieval (txt2img and img2txt) experiment exploiting the CLIP encoded features from MSCOCO and Flickr30k. 

OpenAI’s CLIP (Contrastive Language–Image Pre-training) a Vision-and-Language Transformer (ViLT), a dual-encoder network pre-trained on a corpus of 400 million (image, text) pairs collected from a variety of publicly available sources on the Internet.  It is trained using a Contrastive Loss that takes as input paired text and image embeddings and tries to bring similar pairs closer together and push away dissimilar ones.

MSCOCO (Microsoft Common Objects in Context): a large-scale object detection, segmentation, key-point detection, and captioning dataset. It consists of 328K images each paired with up to 5 natural language descriptions. (2017 validation split)
Flickr30k: contains 31,000 images together with 5 reference sentences provided by human annotators.

data  
├── coco2017
│   ├── annotations  
│   │   ├── train2017_caption.json 
│   │   ├── val2017_caption.json 

│   ├── train2017  
│   │   ├── ####.jpg  
│   ├── val2017  
│   │   ├── ####.jpg  

├── Flickr30k (split in to train,validation)
│   ├── annotations  
│   │   ├── ###.txt    
│   ├── images  
│   │   ├── ###.jpg 
