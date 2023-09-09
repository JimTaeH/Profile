<h1> Hello Everyone, I am DRAGON. <br />Interested in Artificial Intelligence, Science, and Programming </h1>
<b><h2> Let's see my highlights project. </h2></b>
<ul>
  <li> <a href="/#-image-captioning-with-clip-prefix-caption-model-on-traffy-fondue-datasets-/">Image Captioning with CLIP Prefix Caption Model on Traffy Fondue Datasets</a> </li>
  <li> <a href="/#/">Training Wav2Vec 2.0 XLSR-53 on Large Thai Language Datasets and Deploying on Triton Server</a> </li>
</ul>

<h1> Image Captioning with CLIP Prefix Caption Model on Traffy Fondue Datasets <br />
  <a href="https://github.com/JimTaeH/PrefixEmbeddingCLIPCAP"> See detail in this repository, </a>
  <a href="https://drive.google.com/file/d/1jOpQsx04nz31tsQ0x642M1iL0ZcS-Lzg/view?usp=sharing"> Video Explain (Thai), </a>
  <a href="https://drive.google.com/file/d/1mAQwcsGESS_ufrbY9UTGcpcCL_W_Beha/view?usp=sharing"> Paper (Can share only draft version). </a>
</h1>
In this project, the purpose is to generate the captions from images. And use that caption to do some things such as search for images from similar captions or classify the image from its caption. With this hypothesis, our idea is to help Traffy Fondue staff easier to deal with incoming images from users e.g. verifying problems using a caption of an image matching the request from user or classify the problem from caption. With this model, it enables the computer to transcribe image context into text description and use it to process or do something more varied.

<h3> Example of generated caption </h3>
<div align="center">
  <img width="840" hight="460" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/Pic%20CLIP%20CAP%203.png?alt=media&token=f1b6809e-fe9a-487a-aa3d-f15f300b6026">
</div>

<h3> How this model work? </h3>
<p> Refer and thanks to <a href="https://github.com/rmokady/CLIP_prefix_caption">R. Mokady, CLIP_prefix_caption</a>. This model consists of 3 model that is CLIP, Mapping Network (Prefix Caption Model), and GPT-2. Process starts with inputting image into CLIP model for extract features of image. Then using image features to create constant called prefix embeddings by input it into Mapping Network (The Mapping Network is transformer-based mapping network from the CLIP embedding space and a learned constant to GPT-2). After getting the prefix embeddings, feed it into GPT-2 model and let's the model generate caption from it prefix. Finally using GPT-2 decoder to decode the tokens into texts (caption).</p>
<div align="center">
  <img width="840" hight="460" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/Pic%20CLIP%20CAP%202.png?alt=media&token=91bf01ff-9c45-427d-8f71-483041f30c88">
</div>

<h3> Our methodology in this experiments </h3>
<p> In this experiment, we started by collecting and preparing datasets. Collecting the data from #Traffy x TeamChadChart website. They provide CSV files that contain many attributes. We select only "Photo" (image URL) and comment (text prompt by user). Then download all images locally for convenience to read and pass into the model. We will get images and its caption (assumes user comment as caption)</p>
<p> Then create Python virtual environment and install all dependencies <a href="https://github.com/JimTaeH/PrefixEmbeddingCLIPCAP"> detail in this repository </a>.</p> 
<p> And download pretrained model of CLIP, GPT-2 tokenizer, and the Mapping Network (In this work we use only pretrained weights. No fine-tuning because limitation of hardware and data.)</p>
<p> Finally, we have to verify quality of generated caption by metrics BLEU and ROUGE. So we feed 6,001 images into model and generate 6,001 captions. We need to translate user's comment into Thai because model can only generate English caption (without fine-tune). After done everythings we get the result following below table. </p>
<div align="center">
  <h3> Result </h3>
  <img width="695" hight="631" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/messageImage_1694245525728.jpg?alt=media&token=7271d6b7-4b36-4dc0-9487-a4bf7c05575e">
</div>
<p> As you can see in the table, the scores are very low because the user comments (Ground Truth) aren't directly caption of images. So it is hard to determine that the generated caption (Predicted) is correct. With this result, we cannot use it to generate captions and classify problem categories. (And sometimes it generate bad or non related caption.)</p>

<h3> Our Second Approach is using Prefix embedding and doing UMAP to see how it clusters. </h3>
<p> In this approach, we input images into CLIP and send image features into the Mapping Network to create prefix embedded. After this, instead of send it to GPT-2. We reduce dimensional complexity with UMAP and do some standard scaler to prefix embedded. And send this value to Wandb.ai for create   </p>
