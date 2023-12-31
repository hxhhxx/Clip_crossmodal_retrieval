import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #action
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--num_epoch', type=int, default="5",help="num of epoch") 
    parser.add_argument('--batch_size',type=int, default="16",help="Batch size of dataloader")     
    parser.add_argument('--lr',type=float, default="1e-4",help="Optrimizer learning rate") 

    #model&training
    parser.add_argument('--model', default="ViT-B/32",help="Choose the model of convert")
    parser.add_argument("--trainable", default="linear", help="Loss of the model")
    parser.add_argument("--loss", default="cross_entropy", help="Loss of the model")
 
    # datasets parameters
    parser.add_argument("--dataset", default="flickr", choices=["flickr", "coco"], help="Choose the dataset to process (flickr/coco)")
    
    #we need to use different set of moscoco for different tasks
    parser.add_argument("--train_root", default='/kaggle/input/flickr30k/Images', help="Root directory of the dataset")
    parser.add_argument("--train_ann", default="/kaggle/input/flickr30k/captions.txt", help="Annotation file of the dataset")
    parser.add_argument("--test_root", default='/kaggle/input/flickr30k/Images', help="Root directory of the dataset")
    parser.add_argument("--test_ann", default="/kaggle/input/flickr30k/captions.txt", help="Annotation file of the dataset")
    parser.add_argument("--val_root", default='/kaggle/input/flickr30k/Images', help="Root directory of the dataset")
    parser.add_argument("--val_ann", default="/kaggle/input/flickr30k/captions.txt", help="Annotation file of the dataset")
    
    args = parser.parse_args()
    return args