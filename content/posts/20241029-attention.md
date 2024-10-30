+++
title = 'Attention is what we needed'
date = 2024-10-29T21:31:23+01:00
draft = true
tags = ['nlp', 'transformers', 'sequence-to-sequence', 'rnn', 'deep-learning']
+++

It has been demonstrated that transformers can excellently overcome challenges such *NLP*, *Text-To-Image Generation* or *Image Completion*  with large datasets, great model size and enough compute. Talking about transformers nowadays is as casual as talking about *CNNs*, *MLPs* or *Linear Regressions*. Why not take a glance through this state-of-the-art architecture?

## Sequence-to-sequence paradigm

## Recurrent Neural Networks

### Understanding attention

## Transformer architecture

The **transformer neural network** is composed by an encoder-decoder architecture  much like **RNN**. However, the difference is that the input sequence can be passed in parallel by passing also the positional encoder zipped with, as the input might have different meaning depending on its position. Therefore, the positional encoder is a vector that gives context based on position of the element in a sentence. For every odd step, they create the vector using the cosine function while for every even time step, they use the sine function. These functions have linear properties the model can easily learn to attend to when adding these vectors to their corresponding vector. The result of these functions will be concatenated to the vector.

The input and the positional encoding are passed into the encoder block. The job of the encoder is to map all input sequence into abstract continuous representation that holds the learned information for that entire sequence. The encoder block has $$N$$ identical encoder layers. Each layer has two sub-layers: a multi-head attention layer and a feed forward layer. Both sub-layers have a residual connection and a layer normalization next to their output vector. The residual connections helps the network to train by allowing gradients to flow directly through the network while the normalization is used to stabilize the network.

Regarding the decoder block, it has several similarities with the encoder block. They both have $N$ identical layers and a position encoding at first of all. However, multi-head attention layers of the decoder block have different job compared to the encoder. The decoder is auto-regressive and it takes the previous outputs from itself and the encoder output vector as inputs. This is because the encoder can use all the elements of the input sentence  but the decoder can only use the previous elements of the sentence. The attention got in encoder blocks is called **bidirectional attention** while the decoders captures **casual attention**.

### Multi-Head Attention

Multi-head attention block applies a specific attention mechanism called self-attention (in case of encoder), which allows the model to associate each individual element in the input to other elements. The attention is computed by how relevant is the \textit{i}\textsuperscript{th} element of the sentence to other words in the same sentence. This means that it is computing what part of the sentence should the model focus. Therefore, we can understand the attention as a vector that captures the contextual relationships between elements in the sentence. 



#### Scaled Dot Product Attention Head

In other words, the query is a vector related with what we encode, the key is a vector related with what we use as input to output and the value is the learned vector as a result of calculations but related with the input. Regarding encoder block, all three vectors are the source vector since what is wanted is to capture the attention of the elements between them.

#### Multi-Head Attention block


To give the encoder model more representation power of the self-attention, the block is split into several blocks called heads, which can be run concurrently. The input of each head is fed into three distinct fully connected layers to create the query, key and value vectors. The idea is that the attention block must map the query against a set of keys to then present the best attention, which will be embedded to the values. 


#### Masked Multi-Head Attention block

Therefore, the first multi-head attention layer is masked so that it prevent it from being conditioned of the future tokens. In order to do that, once having the score matrix in a head, the result will be masked before calculating the softmax. Regarding the second multi-head attention layer, it can be seen that the queries and the keys are the encoder output while the values are the output of the first attention layer of the decoder, which also are the residual connection. Then, the result is passed into a feed forward layer for further processing and finally, goes to a linear layer that access a classifier and a softmax function to turn it into a probability distribution.
