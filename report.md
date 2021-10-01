### [Representational Learning](https://link.springer.com/chapter/10.1007/978-981-15-5573-2_1)

Natural languages are typical unstructured information. Conventional  Natural Language Processing (NLP) heavily relies on feature engineering, which requires careful design and considerable expertise.
Representation learning aims to learn representations of raw data as  useful information for further classification or prediction.

Feature engineering needs careful design and considerable expertise, and a specific task usually requires customized feature engineering  algorithms, which makes feature engineering labor intensive, time  consuming, and inflexible. Representation learning aims to learn informative representations of objects from raw data automatically.

We train a sequence Transformer to auto-regressively predict pixels, without incorporating knowledge of the 2D input structure.

## Transformers

### Sequence to Sequence Learning and Attention

Sequence-to-Sequence (or Seq2Seq) is a neural net that transforms a  given sequence of elements, such as the sequence of words in a sentence, into another sequence. Seq2Seq models consist of an Encoder and a Decoder.

The attention-mechanism looks at an input sequence and decides at each step which other parts of the sequence are important
For each input that the LSTM (Encoder) reads, the attention-mechanism  takes into account several other inputs at the same time and decides  which ones are important by attributing different weights to those  inputs. The Decoder will then take as input the encoded sentence and the weights provided by the attention-mechanism. To learn more about  attention, see [this article](https://skymind.ai/wiki/attention-mechanism-memory-network). And for a more scientific approach than the one provided, read about  different attention-based approaches for Sequence-to-Sequence models in [this great paper](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf) called *‘Effective Approaches to Attention-based Neural Machine Translation’.*

### Transformers

for each input that the LSTM (Encoder) reads, the attention-mechanism  takes into account several other inputs at the same time and decides  which ones are important by attributing different weights to those  inputs. The Decoder will then take as input the encoded sentence and the weights provided by the attention-mechanism. To learn more about  attention, see [this article](https://skymind.ai/wiki/attention-mechanism-memory-network). And for a more scientific approach than the one provided, read about  different attention-based approaches for Sequence-to-Sequence models in [this great paper](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf) called *‘Effective Approaches to Attention-based Neural Machine Translation’.*

## Aproach

1. pre-training stage
2. fine-tuning stage

## [BERT](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

BERT (Bidirectional Encoder Representations from Transformers) is a recent [paper](https://arxiv.org/pdf/1810.04805.pdf) published by researchers at Google AI Language.
BERT’s key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language  modelling. This is in contrast to previous efforts which looked at a  text sequence either from left to right or combined left-to-right and  right-to-left training. The paper’s results show that a language model  which is bidirectionally trained can have a deeper sense of language  context and flow than single-direction language models.

To overcome this challenge, BERT uses two training strategies:

### Masked LM (MLM)

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token.
The model then attempts to predict the original value of the masked  words, based on the context provided by the other, non-masked, words in  the sequence. In technical terms, the prediction of the output words  requires:



## [Pre-training in NLP](https://medium.com/ai%C2%B3-theory-practice-business/what-is-pre-training-in-nlp-introducing-5-key-technologies-455c54933054)

* Pre-training in AI refers to training a model with one task to help it form parameters that can be used in other tasks.
* using model parameters of tasks that have been learned before to initialize the model parameters of new tasks.

[Paper Explaination](https://becominghuman.ai/image-gpt-generative-image-one-pixel-by-pixel-fbebf784d48)

In general, the pre-training methods are done in language models on the text. But now they are pre-training on image generation.
