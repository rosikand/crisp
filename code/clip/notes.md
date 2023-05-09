Files we might need: 

- transform.py for image preprocessing
- timm_model.py (maybe) 
- modified_resnet.py
- model.py
- loss.py 
- factory.py (maybe)



Also, I think we should create custom types for the classes and call assertions throughout the code to make sure we are correct. See https://github.com/mlfoundations/open_clip/blob/6ee59e10510ec9761b8b9871b9fd1eeb8e28627d/src/open_clip/zero_shot_classifier.py#L40. 


Approaches: 
- inherit CLIP class from open_clip package and modify it to our needs
- or... write our own class from scratch. 


Misc.: 
- locking either image or text encoder improved performance: https://arxiv.org/abs/2111.07991. 


Note: would be good to add in ModifiedResnet class eventually since OpenAI uses that officially https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L94. 


Some insight into the labels: 

<p align='center'>
    <img alt="picture 1" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/81d12243cb2ae3a20143d9d255f7ac8e07692724ea52aa6761d2e42e6be86562.png" width="500" />  
</p>

Basically, get the 10x10 matrix which is just dot products normalized (which is the same as cosine similarity). This will produce [batch_size, 10] shape logits. Now we want to essentially “predict” which cosine sim score is a positive pair (i.e., positive cosine sim consists of the dot product between matching ground level and remote sensing image and negative is non-matching dot product). Thus, the loss is cross entropy which will maximize the cosine sim scores where i ==j (diagonal, like we had discussed)  and minimize the rest which will lead to matching pairs being close together in the 512-dim latent space. 