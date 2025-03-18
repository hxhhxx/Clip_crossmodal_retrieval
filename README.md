# Clip_crossmodal_retrieval

The main objective of this project is to set up a crossmodal retrieval (txt2img and img2txt) experiment exploiting the CLIP encoded features from MSCOCO and Flickr30k（or finetune on your own dataset). 

OpenAI’s CLIP (Contrastive Language–Image Pre-training) a Vision-and-Language Transformer (ViLT), a dual-encoder network pre-trained on a corpus of 400 million (image, text) pairs collected from a variety of publicly available sources on the Internet.  It is trained using a Contrastive Loss that takes as input paired text and image embeddings and tries to bring similar pairs closer together and push away dissimilar ones.

# The Main Idea

The loss function of CLIP is based on the average similarity of image-to-text (img2text) and text-to-image (text2img) which is computed below:

The similarity between two embeddings \( z_i \) and \( z_j \) is defined as:

\[
sim(z_i, z_j) = \frac{z_i^T z_j}{\|z_i\| \|z_j\|}
\]

The CLIP contrastive loss function is defined as:

\[
L_{i,j} = -\log \left( \frac{\exp\left( \frac{sim(z_i, z_j)}{\tau} \right)}{\sum_{k=1}^{2N} 1_{[k \neq i]} \exp\left( \frac{sim(z_i, z_k)}{\tau} \right)} \right)
\]

Here, we employ a similarity pair wise contrastive loss function that directly drives the similarity of correct image-text pairs closer to 1. The function is defined as:

\[
L_{i,j} =
\begin{cases}
\frac{1}{2} (1 - sim(z_i, z_j)) & \text{if } y = 1 \\
\frac{1}{2} \max(0, sim(z_i, z_j) - m) & \text{if } y = 0
\end{cases}
\]

where:

- \( y = 1 \) indicates a positive pair,
- \( y = 0 \) indicates a negative pair,
- \( m \) is a margin parameter.

This modification is more suitable for **smaller batch sizes** and is no longer affected by **temperature** parameters or **resource** constraints.

## Hardware and Training Details

| **Parameter**   | **Value**         |
|-----------------|-------------------|
| GPU             | NVIDIA Tesla P100 |
| Batch Size      | 32                |
                Fig.1

Meanwhile, we freeze and store the other parameters of the model, training only the final layer, achieving high accuracy with rapid speed and limited resources(hardware shown on Fig.1).

And here is the result:
![示例图片](./_img/result.png)

To use the code straightly, try:
```python
!pip install git+https://github.com/openai/CLIP.git
!git clone https://github.com/hxhhxx/Clip_crossmodal_retrieval.git
!pip install pycocotools
!python /kaggle/working/Clip_crossmodal_retrieval/main.py --batch_size "256" --trainable "adaptor"   --dataset "coco" --num_epoch "1" --model "ViT-L/14" # finetune defaultly, if just eval change the para
```