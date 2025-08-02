---
title: Swapping self attention with fourier transforms
date: 10-07-2025
description: Key considerations when building machine learning systems that need to perform reliably in production environments.
tags: ["transformers", "fast inference", "fourier transform"]
---

# Swapping self attention with fourier transforms

## Table of Contents

1. [`Introduction`](#introduction)
2. [`Hypothesis`](#hypothesis)
3. [`Finetuning on IMDB binary classification on GPU`](#finetuning-on-imdb-binary-classification-on-gpu)
   - [`Finetuning results - Vanilla-Bert-Tiny`](#finetuning-results---vanilla-bert-tiny)
   - [`Finetuning results - Fnet-Bert-Tiny`](#finetuning-results---fnet-bert-tiny)
   - [`Time required for training`](#time-required-for-training)
4. [`Finetuning on IMDB binary classification on CPU`](#finetuning-on-imdb-binary-classification-on-cpu)
   - [`Graphical results`](#graphical-results)
5. [`Conclusion`](#conclusion)
   - [`Inference time`](#inference-time)
   - [`Training time`](#training-time)
   - [`Accuracy`](#accuracy)
6. [`Reference`](#reference)

---

## Introduction

This is an experiment to speed up traditional transformer based encoders , with limited accuracy cost, by replacing the self attention layer with 2D fourier transform.  

Self attention provides a way to "mix" the input tokens, so that we can establish relationships between each token of the input. The time complexity for such operation is O(n2). 
There have been many alternatives to reduce this complexity; one such almost radical work is by using fourier transform instead of self attention. This will replace the learnable layer with a fixed (non-learnable) way to "mix" the input tokens. 
For this experiment I will use a 2d transform across 2 dimensions ie. across sequence length and hidden dimenension. This time complexity comes out to be O(nlogn). [FNet- original paper] (https://www.alphaxiv.org/abs/2105.03824v4)


## Hypothesis

1. Reduce finetuning/training time becuase of reduction in total learnable paramaters since unlike self attention we are not learning the weughts (and biases!)
2. Shave off inference time (lets say by ~10 percent) yet being usable (acccuracy ~90%) wrt. the original model.

I understand that the realtive training, inference time and performace will vary based on the implementation of the 2d fourier transform (mainly DFT vs FFT). For this experiment I am using the FFT module present in pytorch. An extension of this expeirment can be to compare different fourier algortihms across devices (GPU, CPU, TPU).  

These are the assumptions/tools I will be relying on to prove (or not !) this hypothesis:

- using a small embedding model like bert-tiny
- only swap the self attenions layer with 2d fourier transform, keeping the rest identical to the base model ie. bert-tiny
- There can be many implemetation to do fourier transform . I will rely on the torch implementation for FFT (fast fourier transform)
- Intuitively the fouruier transform will decompose the input signal (seq length , hidden dimension) into a combination of known signals , which in turn will be passed on the linear layer following it. 
- Simple NLP task like binary classification - imdb 
- There are many implemetations of doing a 2d fourier transform. For this experiment I will be using this implemetation inspired from the orignal Fnet paper.
```python
class FNetBlock(nn.Module):
"""The FNet block which replaces the self-attention layer."""
def __init__(self):
    super().__init__()

def forward(self, hidden_states, attention_mask=None, head_mask=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            past_key_value=None, output_attentions=False):
    # Apply 2D FFT and take the real part. The output is a single tensor.
    return (torch.fft.fft(torch.fft.fft(hidden_states, dim=-1), dim=-2).real,)
```
The first FFT is applied across the sequence length and the second transform is applied across the hidden dimension, and only the real part of the final output is considered. 

##  Finetuning on IMDB binary classification on GPU

Imdb is a small dataset for classifying sentiments (positive or negative) based on a text. We will use this data to test out the hypothesis. 

Throught the experiment , the validation loss is consistantly going up after the 6th/7th epoch, however for brevity I will only choose the best performing model based on accuracy irrespective of it midly overfitting on the train data. 

### Finetuning results - Vanilla-Bert-Tiny
![Vanilla-Bert-Tiny - Train, validation loss and accuracy on finetuning on the IMDB dataset](/static/images/blog/1/1.1.png)*Vanilla-Bert-Tiny trained on GPU - Train, validation loss and accuracy on finetuning on the IMDB dataset.*

### FInetuning results - Fnet-Bert-Tiny
![Fnet-Bert-Tiny - Train, validation loss and accuracy on finetuning on the IMDB dataset ](/static/images/blog/1/1.2.png)*Fnet-Bert-Tiny trianed on GPU - Train, validation loss and accuracy on finetuning on the IMDB dataset*

## Time required for training

![Fnet-Bert-Tiny and bert-tiny (vanilla) - Training and inference time](/
static/images/blog/1/1.3.png)*Fnet-Bert-Tiny and bert-tiny (vanilla) on GPU - Training and inference time*

Similary to Vanilla-Bert-Tiny finetuning, the Fnet variant also exhibits spike in validation loss after the 3rd epoch.

## Finetuning on IMDB binary classification on CPU


![Vanilla-Bert-Tiny - Train, validation loss and accuracy on finetuning on the IMDB dataset](/static/images/blog/1/1.4.png)*Vanilla-Bert-Tiny trined on CPU - Train, validation loss and accuracy on finetuning on the IMDB dataset*


![Fnet-Bert-Tiny - Train, validation loss and accuracy on finetuning on the IMDB dataset](/static/images/blog/1/1.5.png)*Fnet-Bert-Tiny trianed on CPU - Train, validation loss and accuracy on finetuning on the IMDB dataset*

![Fnet-Bert-Tiny and bert-tiny (vanilla) - Training and inference time](/
static/images/blog/1/1.6.png)*Fnet-Bert-Tiny and bert-tiny (vanilla) on CPU - Training and inference time*

### Graphical results 
![Fnet-Bert-Tiny vs bert-tiny (vanilla) - Training and inference time](/
static/images/blog/1/1.7.png)*Fnet-Bert-Tiny vs bert-tiny (vanilla)*

![Accuracy vs inference speed by model and device](/
static/images/blog/1/1.8.png)*Accuracy vs inference speed by model and device*

### Conclusion

#### Inferece time

- on CPU fnet-bert-tiny is ~43% faster than vanilla-bert-tiny
- on GPU fnet-bert-tiny is ~24% faster than vanilla-bert-tiny

#### Training time

- on CPU fnet-bert-tiny trains ~33% faster than vanilla-bert-tiny
- on GPU fnet-bert-tiny trains ~9% faster than vanilla-bert-tiny

#### Accuracy

- on CPU fnet-bert-tiny achieves ~96% accuracy of vanilla-bert-tiny
- on GPU fnet-bert-tiny achieves ~96% accuracy of vanilla-bert-tiny

### FNet variations

different implemtation of fourier transformns in pytorch - https://docs.pytorch.org/docs/stable/fft.html

### Reference

1. [FNet: Mixing Tokens with Fourier Transforms] (https://www.alphaxiv.org/abs/2105.03824v4)
