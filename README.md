This is our GNR-638 project.

## Work Flow

The project can be divided into 3 parts broadly :-

1. Preprocessing the image  : - I was unable to understand the exact details of the preprocessing in the paper.  " First, we pre-process raw images by resizing to a low resolution and reshaping into a 1D sequence. " **Someone can start working on this part, try to collect some data, use opencv to cut the images (To test initially we can use a few images but later we would need to collect more images. Try to collect images related to iitb as a proof that our model works). Figure out how OpenAI did preprocessing in this paper.**
2. **Pre-training** :- This part mainly involves the BERT encoder. I have made some reference in dev branch. This is an NLP model.  **Someone can start studying about this model and try to implement it. Its not possible to judge the model right now. We can do integration part later sometime**
3. **Architecture** :- This is the GPT-2 decoder part. **Someone can start working on this and try to write it (Its important to colaborate with the person working BERT model here)**
4. Fine-tuning :- **I think this part is not too big**
5. Linear Probing ;- **As far As I understood this is used as a metric for judging the model. I guess it can be made optional for our project.**

Final Training would require making changes to BERT and GPT-2 models so, training etc can be done after Endsem :) If deadline extends :'( .

PS: The main challenge is to understand whats going on. I honestly didn't get how and when output would be obtained but I expect we would understand in the process :). 

**Feel Free To makes changes to this README**  
### Updates  
(Neeraj) : I have completed the coding part of the model (i.e. steps 2+3 from the sketch above). Note that BERT (masking some pixels randomly while training instead of predicting one-by-one) was not used so far. I used the GPT-2 architecture currently.  
Future work will be  
- To train it neatly on the preprocessed dataset  
- Fine tune the trained model for classification (take output of any decoder block and retrain this model for classification)
- Linear probing is just a way to gauge the performance of different output layers as a feature vector for logistic regression model.  

Link (notebook) : https://colab.research.google.com/drive/148yiVLIHWIv9r9GNDLALyG-9EgL7aUFT#scrollTo=nHc7YALSuS9C
