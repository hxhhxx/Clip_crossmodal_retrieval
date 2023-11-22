import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # datasets parameters
    parser.add_argument("--dataset", default="flickr", choices=["flickr", "coco"], help="Choose the dataset to process (flickr/coco)")
    # parser.add_argument("--root", default='/kaggle/input/flickr30k/Images', help="Root directory of the dataset")
    # parser.add_argument("--ann_file", default="/kaggle/input/flickr30k/captions.txt", help="Annotation file of the dataset")
    parser.add_argument('--evaluate', action='store_true')
    
    args = parser.parse_args()
    return args