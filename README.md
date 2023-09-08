<h1> Hello Everyone, I am DRAGON. <br />Interested in Artificial Intelligence, Science, and Programming </h1>
<b><h2> Let's see my highlights project. </h2></b>
<ul>
  <li> <a href="/#-image-captioning-with-clip-prefix-caption-model-on-traffy-fondue-datasets-/">Image Captioning with CLIP Prefix Caption Model on Traffy Fondue Datasets</a> </li>
  <li> <a href="/#/">Training Wav2Vec 2.0 XLSR-53 on Large Thai Language Datasets and Deploying on Triton Server</a> </li>
</ul>

<h1> Image Captioning with CLIP Prefix Caption Model on Traffy Fondue Datasets </h1>
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
<p> In this experiments, we starting from  </p>
<!-- <div align="center">
  <img width="840" hight="460" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/Pic%20CLIP%20CAP%202.png?alt=media&token=91bf01ff-9c45-427d-8f71-483041f30c88">
</div> -->
