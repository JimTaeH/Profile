<h1> Hello Everyone, I am DRAGON. <br />Interested in Artificial Intelligence, Science, and Programming </h1>
<b><h2> Let's see my highlights project. </h2></b>
<ul>
  <li> <a href="/#-image-captioning-with-clip-prefix-caption-model-on-traffy-fondue-datasets-/">Image Captioning with CLIP Prefix Caption Model on Traffy Fondue Datasets</a> </li>
  <li> <a href="/#-training-wav2vec-20-xlsr-53-on-large-thai-language-datasets-and-deploying-on-triton-server-/">Training Wav2Vec 2.0 XLSR-53 on Large Thai Language Datasets and Deploying on Triton Server</a> </li>
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
<p> Refer and thanks to <a href="https://github.com/rmokady/CLIP_prefix_caption">R. Mokady, CLIP_prefix_caption</a>. This model consists of 3 model that is CLIP, Mapping Network (Prefix Caption Model), and GPT-2. Process starts with inputting image into CLIP model for extract features of image. Then using image features to create constant called prefix embeddings by input it into Mapping Network (The Mapping Network is transformer-based mapping network from the CLIP embedding space and learned a constant to GPT-2). After getting the prefix embeddings, feed it into GPT-2 model and let's the model generate caption from it prefix. Finally using GPT-2 decoder to decode the tokens into texts (caption).</p>
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
<p> In this approach, we input images into CLIP and send image features into the Mapping Network to create prefix embedded. After this, instead of send it to GPT-2. We reduce dimensional complexity with UMAP and do some standard scaler to prefix embedded. And send this value to Wandb.ai for create clustering visualization</p>
<div align="center">
  <h3> Process for create prefix embeddings for visualize embedding projector </h3>
  <img width="840" hight="460" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/ClipCapEmbedding2.png?alt=media&token=33782d70-9581-4a90-89de-9ac77a4fc820">
</div>
<div align="center">
  <h3> Embeddings Projector: Example of clusters </h3>
  <img width="840" hight="460" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/messageImage_1694251662904.jpg?alt=media&token=29831c33-1ad9-45f4-9197-8cf4c0757691">
</div>
<div align="center">
  <h3> Example of similar images in the same groups of promblem </h3>
  <img width="840" hight="460" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/WandB-Embedding_Projector1.png?alt=media&token=bf7de463-aaea-4372-a51d-a3d15e71b10c">
</div>
<h3> Conclusion </h3>
<p> It is not good enough to do the explicit clusters of images by using only prefix embeddings (as you can see in the embeddings projector). So, in my opinion, we need to fine-tune model for a better result or suitable for Thai context and caption. And suggestion for this work, In terms of actual implementation guidelines, it may start by taking image data that the system already has and creating Embedding Spaces first so that the system has clusters of information and categories. Then when new images are input Those new images are converted into values for Embedding and compared to existing data groups to determine what category the new images fall into, etc. This should help Traffy Fondue to group and classify problems from the images. </p>

<h1> Training Wav2Vec 2.0 XLSR-53 on Large Thai Language Datasets and Deploying on Triton Server </h1>
<p> In this project, I am working on fine-tuning Wav2Vec 2.0 XLSR-53 and deploying it on Triton Server, and making API service using FastAPI with Docker. This is almost end-to-end project for the ASR system! </p>
<ul>
  <b> Overview </b>
  <li> Fine-tuning Wav2Vec2 </li>
    <ul>
      <li> EDA and Cleaning Datasets </li>
      <li> Convert datasets into PyArrow format </li>
      <li> Create tokenizer and feature extractor </li>
      <li> Create features for input to Model </li>
      <li> Download and config pretrained model </li>
      <li> Train, Validation, and Save checkpoint </li>
      <li> Resume training </li>
      <li> Create n-grams language model to improve performance </li>
    </ul>
  <li> Make an API Service </li>
    <ul>
      <li> Prerequisite </li>
      <li> Convert model to ONNX format </li>
      <li> Deploying on Triton Server </li>
      <li> Create API with FastAPI + Docker (also implement VAD and noise reduction) </li>
      <li> Example: Transcribe via audio file upload </li>
      <li> Realtime speech transribe </li>
      <li> Example: Transcribe via input from microphone </li>
    </ul>
</ul>
<h2> Fine-tuning Wav2Vec2 </h2>
<h3> EDA and Cleaning Datasets (Cannot share related code and data)</h3>
<p> In this EDA part, I am just checking the duration of audio files in datasets to see their distribution. To get the duration of audio I can use size of the audio array divided by its sample rate. These NECTEC datasets have the lowest duration at 0 seconds and the highest is about 4:26 minutes. And the transcription consists of Thai and English characters also some digits. </p>

<ul> 
  In the cleaning part I am doing following this list 
  <li> Delete the error audio files, fix mispell or wrong word (if see), fix number </li>
  <li> Select only audio that has the durations of 10-30 seconds </li>
  <li> Fix Thai word that using ๆ </li>
  <li> Check abbreviation </li>
  <li> Clean special characters </li>
  <li> Verify correctness of dataset </li>
</ul>
<p> <ins>Delete the error audio files, fix mispell or wrong word (if see), fix number.</ins> First, the error audio files meaning audio with no transcription or audio with no sound nor speech. I am drop all that files from the dataset. Then I print all transcription and check the spelling and replace the misspelled or wrong words with correct words. (This is something done manually but grateful this dataset has been cleaned beforehand). And fixing the number is to rewrite to related it's pronounce. For example 10 . 30 น. This should be fixed to สิบ นาฬิกา สามสิบ นาที same as it is pronounced. </p>
<p> <ins>Select only audio that has the durations of 1-30 seconds.</ins> Facebook research recommends using audio that has durations of 10-30 seconds. So I keep only audio file that has a duration of 1-30 seconds because I want to keep the amount of audio files in the dataset (with this selection the dataset lose audio files only 1,516 files from many hundred thousand files). And model still works with no problem because Wav2Vec2 need to do masking for every 25ms. So audio 1 second is still fine. In this part, I will get a dataset that has attributes of a path to audio and transcription which audios have a duration in range of 1 to 30 seconds. </p>
<p> <ins> Fix Thai word that using ๆ </ins> The word that uses ๆ must be fixed to its full form. Example, ดีมาก ๆ --> ดีมากมาก. I can check ๆ by using Regular Expression but in the replace process I did it manually to make sure that word will correctly. </p>
<p> <ins> Check abbreviation </ins> I need to verify the abbreviation and whether it pronounces that is related or not. For example, if พ.ศ. in transcription and in the audio file said พ.ศ. this should be fine. But if it said พุทธศักราช this should be fixed. </p>
<p><ins> Clean special characters </ins> For a special characters that doesn't have the pronounces I will remove it all. Example !, ?, @, *, # etc. </p>
<p><ins> Verify correctness of dataset </ins> Finally, checking audio files and transcription again to make sure that haven't any error anymore. </p>

<h3> Convert datasets into PyArrow format </h3>
<p> In this part, I will split the dataset into train and validation sets (we already have another test set). To separate dataset, I shuffle it and cut it into 90% for train and 10% for validation. Then convert it into .arrow format (Python Apache Arrow). This will be able to read and write data with the Datasets library which is suitable for large datasets (low RAM required, fast and convenient when processing data). </p>

<h3> Create tokenizer and feature extractor </h3>
<p> Creating the tokenizer, I am using all characters in datasets which is Thai and English characters and number as a token. And adding [PAD] for padding sentences to the same length, [UNK] for unknown words, and | replace for " " (whitespaces). And feature extractor, I am create by following this </p>
<code>feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                             sampling_rate=16000,
                                             padding_value=0.0, 
                                             do_normalize=True, return_attention_mask=True) </code>
<ul> <br /> Reference from <a href="https://huggingface.co/blog/fine-tune-wav2vec2-english"> Hugging Face Blog Post </a> 
  <li> feature_size: Speech models take a sequence of feature vectors as an input. While the length of this sequence obviously varies, the feature size should not. In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal </li>
  <li> sampling_rate: The sampling rate at which the model is trained on. </li>
  <li> padding_value: For batched inference, shorter inputs need to be padded with a specific value </li>
  <li> do_normalize: Whether the input should be zero-mean-unit-variance normalized or not. Usually, speech models perform better when normalizing the input </li>
  <li> return_attention_mask: Whether the model should make use of an attention_mask for batched inference. </li>
</ul>

<h3> Create features for input to Model </h3>
<p> The audio data must be converted to input values for use as input to the model. Also, transcription must be converted to label ids for using as answers for calculating loss and WER, CER. To convert the datasets I have to read all audio files and keep it array and sample rate in the datasets. Then using feature extractor to extract the input values and keep them in datasets. And transcription uses tokenizer to convert. </p>

<h3> Download and config pretrained model </h3>
<p> Load pretrained model Wav2Vec2 from Hugging Face Hub using following code </p>
<code>
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained(processor_path)
<br/>
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained(
    pretrained_model_path,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()
model.config.ctc_zero_infinity = True
</code>
<p> The code same as the Hugging Face blog post except for model.gradient_checkpointing_enable() and model.config.ctc_zero_infinity = True. because these two configs help to reduce memory usage when training and protect vanishing loss. </p>

<h3> Train, Validation, and Save checkpoint (May not share code here...) </h3>
<p> In this part, I am training the Wav2Vec2 model with PyTorch using DataParallel method. Before training and validation loop I have to prepare following this </p>
<ul>
  <li> Load prepared datasets and set format to torch tensor. </li>
  <li> Prepared collator function for dynamic padding to input values and label ids. </li>
  <li> Using PyTorch dataloader to wrap datasets and collator function together and set batch size. </li>
  <li> Prepared optimizer and scheduler (to control learning rate). </li>
  <li> Load WER and CER metrics. </li>
</ul>
<p> When prepared all things successfully, I started writing a training and validation loop. So the first loop is to specify how many epochs I want to train. Inside it will have a training and validation loop. Each loop enumerates on batch of train and validation datasets. At the end of each epoch I will check if model has better scores on CER (Characters Error Rate) and WER (Words Error Rate) then save it checkpoint using </p>
<code> torch.save({'epoch': EPOCH,
                    'best_epoch': num_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'CER': best_cer,
                    'WER': best_wer}, ckp_path)
</code>

<h3> Resume training </h3>
<p> Just load the model and optimizer state dict and set how many epochs to continue. Then follow the same steps and code in the first train and validation loop. </p>

<h3> Create n-grams language model to improve performance </h3>
<p> You can follow this link to see example and how to <a href="https://colab.research.google.com/drive/1Wvl50XiA9-gVlr2snQEuPC-3GRbRxOGf?usp=sharing"> Create N-grams LanguageModel (Colab) </a>. The language model that we get from this part is in .arpa format which is building using KenLM program. We can convert this to .bin for more memory savings and compatible to run on any environment. And this LM can work with Wav2Vec2 when decoder by receiving the logits output into LM to improve the correctness of transcription (By the way it is just in terms of word frequency).</p>

## Make an API Service
### Prerequisite
We must install all of the following on the host machine to create API with docker and enable the GPU.
* [Docker](https://docs.docker.com/engine/install/) Select your OS and following the instruction.
* [Docker Compose](https://docs.docker.com/compose/install/) follow the instruction.
* [Nvidia Drivers](https://www.nvidia.com/download/index.aspx) Select your GPU and OS specification in the drop-down menu. Then download and install it following by instructions. But in case Windows OS + WSL2 with Ubuntu installed. You must install the Nvidia driver on your Windows system only. Don't install it in WSL2 because WSL2 basically sees the Nvidia driver in Windows.
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) Follow this documentation (But you can skip that setting up docker because we already installed that.)
* [Download PyTorch (.whl) (Optional)](https://download.pytorch.org/whl/torch/) Because sometimes it has a network error when installing Torch which is a big file size. To fix it I recommend downloading the .whl file and installing from locally. (Version: torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl)

### Convert Model to ONNX format
ONNX is the Open Neural Network Exchange Format which is designed to represent any type of Machine Learning and Deep Learning model. With this ONNX our model will be easy to deploy on any machine. And PyTorch has a function that support to convert model to ONNX called ```torch.onnx.export``` [Example here](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

### Deploying on Triton Server
First, What is Triton Server? Triton Inference Server is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. [More detail here](https://github.com/triton-inference-server/server).  
To deploy model on Triton Server we need to have NVIDIA Container Toolkit which requires NVIDIA Drivers. Then prepared ONNX model and config file that was written in Triton Server format. Also preprocessing and postprocessing can add to Triton Server. All of models and pre/post processing we have on Triton it called ensemble models. The Triton Server can receive and respond to requests with HTTP or GRPC. Which is GRPC faster at inference than HTTP because it uses protocol buffers (Protobuf) that simple, small, and faster than JSON or XML.

### Create API with FastAPI + Docker (also implement VAD and noise reduction)
<div align="center">
  <p> How this API work </p>
  <img width="840" hight="460" align="center" src="https://firebasestorage.googleapis.com/v0/b/second-try-cb-pirwud.appspot.com/o/Slide1.jpg?alt=media&token=b3ef65c0-ebc7-4416-a015-b44f774001ea">
</div>   

This is a simple API for transcribing audio files into text transcription. Start by user upload audio file to the client. Then client will keep audio file as bytes data to keep it in memory (in RAM). Using torchaudio to open audio data. Client will get array of audio data and sample rate. If sample rate of audio more than 16kHz client will reset to 16kHz. Finally, send all data to Triton Server for inference.  

(Optional) if user want to upload large audio file e.g. file that has duration more than 30 minutes. Client can do VAD (Voice Activity Detection) to split into small part. And iterable send these small part to Triton Server (Using this model [Silero VAD](https://github.com/snakers4/silero-vad/tree/master)).   
(Optional) if audio file have noise. Client can do noise reduction before send audio data to inference server. using noisereduce library (know this because this blog [สู่ความเงียบสงัด EP.2: เริ่มต้นเขียน Noise Cancellation อย่างง่ายด้วยภาษา Python](https://medium.com/super-ai-engineer/%E0%B8%AA%E0%B8%B9%E0%B9%88%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B9%80%E0%B8%87%E0%B8%B5%E0%B8%A2%E0%B8%9A%E0%B8%AA%E0%B8%87%E0%B8%B1%E0%B8%94-ep-2-%E0%B9%80%E0%B8%A3%E0%B8%B4%E0%B9%88%E0%B8%A1%E0%B8%95%E0%B9%89%E0%B8%99%E0%B9%80%E0%B8%82%E0%B8%B5%E0%B8%A2%E0%B8%99-noise-cancellation-%E0%B8%AD%E0%B8%A2%E0%B9%88%E0%B8%B2%E0%B8%87%E0%B8%87%E0%B9%88%E0%B8%B2%E0%B8%A2%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2-python-e598304cee5f))   

Triton Server receives data from the client and starts preprocessing to create input values. Then feed input values into model to calculate output. Finally send output to postprocessing for decode-to-text transcription and response back to the client. Client will send text transcription to the user.
### Example: Transcribe via audio file upload
This is the first audio file in video.  

https://github.com/JimTaeH/Profile/assets/60215727/473666cb-dfb7-4d60-a2cd-42458cacd477  

This is the second audio file in video.  

https://github.com/JimTaeH/Profile/assets/60215727/fdee1ce5-e604-4fb1-8dda-852dd8ecee9a  

This is the third audio file in video.  

https://github.com/JimTaeH/Profile/assets/60215727/5a3c846b-e8d5-4f24-bb80-067200c8c27b  

Demo video  

https://github.com/JimTaeH/Profile/assets/60215727/2dc8c1a8-ed5d-4863-93d8-699cfb0c6c81  


### Realtime speech transribe
I am using pyaudio to create streaming input from microphone. By receiving audio chunks and keeping them in frames. With my config 1 frame will have 3 seconds of audio. Then sending frames to Triton Server to transcribe and responses in realtime. However, the model in this video is the first version so the performance looks poor. VAD can also help when doing real time transcribing but it is still under implementation.

### Example: Transcribe via input from microphone


https://github.com/JimTaeH/Profile/assets/60215727/3e9c1995-81c5-482e-84bb-b92a470e9c59   

# Thanks for Reading. I will share more project and update more detail in the future. =)

