+++
title = 'Late Interaction Hopfield-Fenchel-Young Network for Sparse Retrieval of Language Model Hidden States'
date = 2026-11-11T17:46:23+01:00
draft = false
tags = ['deep-learning', 'associative-memories', 'hopfield-networks', 'hopfield-fenchel-young', 'sparse-functions', 'nlp', 'transformers', 'attention']
metaDescription = 'Exploring training-free Hopfield-Fenchel-Young networks with late interaction for sparse retrieval and direct integration of LLM hidden-state memories.'
recommendations = ['20241029-attention', '20260410-loss-functions-activations']
[cover]
image = "/posts/2026/hfy-late-interaction/portrait.png"
+++

This work investigates the integration of **Late Interaction** and **Hopfield-Fenchel-Young** (HFY) Networks for
associative
retrieval and direct incorporation of external memories into large language models (LLMs). Unlike conventional retrieval
approaches that compress sequences into single embeddings or provide retrieved documents as textual context, the
proposed architecture stores complete sequences of contextualized token-level hidden states extracted from intermediate
Transformer layers and allows retrieval.

The proposed HFY network exposes how MaxSim can be incorporated into a HFY energy for matrix-valued memories
and how hidden states from decoder models can be used as memories to provide neuronal information retrieval.
Specifically, the experiments were done with hidden states extracted from Qwen3.5-4B and investigate whether
retrieved associate memories can be ingested directly into the model's internal representations to support
context-grounded generation in one full language model inference. In particular, **HFY-based retrieval improves the
identification of relevant factual information** compared with the unaugmented model and dense softmax-based memory
integration in the evaluated example.

## 1. Introduction

**Large language models (LLMs)** acquire extensive knowledge during pretraining, but their ability to access, update,
and reliably use information beyond their parameterized representations remains limited. **Retrieval-Augmented
Generation (RAG)** addresses this limitation by incorporating external information into the generation process,
typically retrieving relevant documents and inserting their textual content into the model's context window. Although
effective, this approach requires a) processing the retrieved information as natural language, increasing the number of
input tokens and repeatedly encoding information that may already exist in a continuous representation; and b) **the
usage of an embedding model external to the generative language model** to embed the memories and query to the vector
database.

An alternative is to represent external memories directly within the LLM's hidden-state space. Instead of retrieving
textual passages, a memory system could store contextualized representations extracted from intermediate Transformer
layers and retrieve and inject them into the model's computation. However, decoder models are not trained to capture
semantically as embedding encoder models do and repeated encoding information would persist in RAG.

Therefore, **two main challenges emerge**: how to retrieve relevant information from high dimensional token-level
representation without repeating key-values information and how to incorporate these
representations into the LLM without disrupting its generation capabilities.

This work explores a **Late Interaction Hopfield-Fenchel-Young architecture** that combines token-level similarity with
sparse associative memory selection. The proposed method stores intermediate hidden-state sequences from a decoder-only
LLM, retrieves relevant information using MaxSim-based similarity and hierarchical \(\alpha\)-entmax weighting, and
constructs an associated representation that is directly injected into the model's intermediate hidden states.

To the best of my knowledge, this is the first investigation of training-free, late-interaction HFY-inspired associative
retrieval over pretrained decoder-only LLM hidden states, with direct integration of the retrieved representations into
the same model's generation process.

### 1.1. Modern Hopfield Networks

**Modern hopfield newtorks** [1] are a class of associative memory models that store information as high-dimensional
vectors
and retrieve it by comparing a query against those stored memories. The idea is that, given a query
\(q \in \mathbb{R}^{d}\) that may be incomplete, noisy or semantically related to stored information, a Modern Hopfield
Network retrieves an associated memory \(M \in \mathbb{R}^{m \times d}\) by evolving the query toward a low-energy
state.
Modern Hopfield Networks are particularly interesting because their standard retrieval update is mathematically
equivalent to the softmax attention mechanism used in Transformers, under an appropriate parameterization. At each step
\(i\), the hopfield network calculate similarities between the \(q_{i}\) and each memory:

\[s_i = \beta m_j^T q_{i}, \qquad \forall m_j \in M \]

Here, \(\beta\) is an inverse-temperature parameter that controls the sharpness of retrieval. Then, the
weights are calculated using an activation function like softmax:

\[p_i=\frac{\exp(s_i)}{\sum_j\exp(s_j)}\]

And retrieve the weighted combination of the memories.

\[q_{i+1} = \sum_i p_i m_i = M^T p\]

Modern Hopfield Networks are **energy-based models (EBMs)**. Instead of explicitly learning which memory to retrieve,
the model defines an energy landscape whose minima correspond to stable associative states. The energy funciton
introduced by [1] is:

\[E(q)=\frac12\|q\|^2-\frac{1}{\beta}\log\sum_{i=1}^{N}e^{\beta m_i^\top q}+C\]

Consider standard scaled dot-product attention:

\[\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\]

Comparing this expression with a Hopfield update:

\[\operatorname{Hopfield}(q)=M^\top\operatorname{softmax}(\beta Mq)\]

reveals that both mechanisms compute similarity-based weights and aggregate stored representations accordingly.
When the keys and values correspond to the same memory patterns, and the scaling parameters are appropriately matched,
the two operations become equivalent.

### 1.2. Hopfield-Fenchel-Young (HFY) Networks

Although softmax provides a differentiable and computationally effective mechanism, it assigns **strictly positive
weights** to every stored memory, regardless of how weakly a memory is related to the query.

Consequently, the retrieved representation always contains contributions from all stored patterns. In associative memory
systems containing many unrelated or highly similar memories, these contributions may **introduce interference** and
produce representations that blend multiple memory patterns rather than accurately recovering an individual one.
However, this interference is not inevitable, since softmax can still produce highly concentrated distributions.

To address this limitation, [2, 3] introduced **Hopfield-Fenchel-Young (HFY) Networks**,
a generalized framework for associative memory retrieval based on Fenchel-Young losses and convex regularization.

\[ p= \underset{p\in\Delta_m}{\operatorname{argmax}} \, \{ p^\top s-\Omega(p)\} \]

where \(\Delta_m\) denotes the probability simplex, \(s=\beta \, Mq\) contains the memory similarity scores, and
\(\Omega\) is a convex regularization function. The purpose of \(\Omega\) is to balance two objectives: selecting
memories with high similarity scores and controlling how concentrated or distributed the retrieval weights become.

Different choices of \(\Omega\) lead to different retrieval transformations, including softmax, sparsemax, and
\(\alpha\)-entmax. For the \(\alpha\)-entmax family, [3] use the Tsallis \(\alpha\)-negentropy:

\[ \Omega_\alpha(p)= \frac{\sum_{j=1}^{m}p_j^\alpha-1} {\alpha(\alpha-1)},\qquad \alpha>1 \]

for \(p\in\Delta_m\). At \(\alpha=1\), it is defined by its continuous limit:
\[ \Omega_1(p)=\sum_{j=1}^{m}p_j\log p_j \]

The resulting retrieval transformation is:
\[ p^*=\operatorname{entmax}_\alpha(\beta Mq)\]

The parameter \(\alpha\) controls the sparsity of the memory weights. When \(\alpha = 1\), softmax is used; when
\(\alpha < 2\), ent-max is used and zero-weights can be assigned; when \(\alpha = 2\), sparsemax is used; and when
\(\alpha > 2\), \(\alpha\)-en is used and it produce increasingly selective distributions for fixed scores.

### 1.3. Late Interaction Similarity

Late interaction and HFY associative retrieval solve different problems. HFY addresses how stored memories are weighted
and retrieved; late interaction addresses how the query and stored sequences are compared. Specifically, late
interaction defines how similarities between token-level sequences are computed, preserving
fine-grained alignment between a query and stored sequences. Introduced by Khattab and Zaharia [4] in ColBERT, late
interaction differs from conventional dense retrieval approaches that compress an entire sentence into a single
embedding. Instead, each sequence is represented by a matrix of contextualized token embeddings, allowing similarities
to be calculated at the token level. Figure 1 shows the comparison between
sentence embedding models and late interaction models.

![alt text](/posts/2025/late_interaction.png "Fig 1. Side-by-side comparison of single vector comparison (left) and late interaction (right) models.")

For a query \(Q\in\mathbb{R}^{T_q\times d}\), late interaction provides a sequence-level similarity:

\[ s_j(Q)=\frac{1}{T_q}\sum_{i=1}^{T_q} \max_t \operatorname{sim}(Q_i,M_{j,t})\]

where \(T_q\) and \(T_j\) denote the number of query and memory tokens, respectively, and
\(\operatorname{sim}(\cdot,\cdot)\) measures the similarity between two token representations. In the current
implementation, normalised cosine similarity is used:

\[
\operatorname{sim}(Q_i,M_{j,t}) =
\frac{Q_i^\top M_{j,t}}
{\lVert Q_i \rVert_2 \, \lVert M_{j,t} \rVert_2}
\]

For each query token, the maximum similarity across all tokens in the stored memory is selected. These maximum
similarities are then averaged across query tokens to obtain a single sequence-level similarity score. This operation,
commonly known as **MaxSim**, preserves token-level matching information during similarity computation
without requiring the entire sequence to be compressed into a single pooled embedding.

Unlike single-vector similarity, late interaction can **capture multiple semantic correspondences** between a query and
a
stored sequence, making it particularly suitable for comparing contextualized token representations extracted from
language models.

### 1.4. Related Work and Contributions

Rather than providing an exhaustive literature review, this section highlights three colsely related approaches to
integrating external knowledge into large language models through continuous representations, followed by the main
distinctions of the proposed method.

**xRAG** [5] introduces a modality bridge that maps retrieved document embeddings into an LLM's representation space,
enabling
knowledge integration without including the original textual documents. However, its reliance on compressed
document-level embeddings does not preserve the full sequence of token-level representations. **Hippocampo-neocortical
interaction as compressive retrieval-augmented generation** [6], published in Nature Communications, proposes a
computational model combining modern Hopfield networks with LLM-based generation.
The approach uses associative and sequential retrieval mechanisms to recover compressed episodic memories, which are
subsequently reconstructed by a generative model. Unlike the current approach, it focuses on compressed memory traces
and their reconstruction, rather than preserving and retrieving complete sequences of contextualized LLM hidden states
through token-level late interaction and HFY-based selection. **MHA-RAG** [7] represents retrieved exemplars as learned
soft prompts using multi-head attention. While this enables efficient knowledge integration without conventional textual
demonstrations, it does not employ HFY-based associative
retrieval over stored token-level memories. **Deep Fusion Memory** [8] integrates learned memory representations into
intermediate LLM hidden states through gated cross-attention. Unlike retrieval-based approaches, it predicts memory
representations directly from the
preceding context, eliminating explicit retrieval during inference.

## 2. Late Interaction Hopfield-Fenchel-Young Network

Building upon these directions, this work investigates **Late Interaction Associative Memories with
Hopfield-Fenchel-Young Networks**, focusing on three main aspects:

1. **Token-level memory preservation**: Rather than compressing documents or sentences into individual embeddings, the
   system stores complete sequences of contextualized token representations extracted from intermediate LLM layers,
   preserving fine-grained information.
2. **Associative retrieval through HFY networks**: Token-level late interaction are combined with HFY-based sparse
   memory
   selection, exploring whether structured token similarities can improve associative retrieval without requiring
   sequence
   pooling.
3. **Direct integration of retrieved representations**: Instead of reconstructing retrieved memories into their original
   textual form, injecting retrieved token-level representations are directly fed into intermediate LLM hidden
   states to support context-grounded generation (Figure 2).

Together, these components investigate whether sequence-preserving associative retrieval can provide an alternative to
compressed embedding-based memories and conventional text-based retrieval-augmented generation.

![alt text](/posts/2026/hfy-late-interaction/hfy_llm.png "Fig 2. Architecture of the proposed Late Interaction Hopfield-Fenchel-Young (HFY) memory integration into a decoder-only LLM. Intermediate hidden states extracted after \(L_{\mathrm{pre}}\) Transformer layers are used as queries for the HFY memory module. The retrieved token-level representations are concatenated with the original hidden states and processed by the remaining \(L_{\mathrm{post}}\) Transformer layers to generate the output, enabling direct memory integration without reconstructing textual context.")

### 2.1. Late Interaction Associative Memory Integration

The proposed architecture integrates a Late Interaction HFY memory module into an intermediate
layer of a decoder-only LLM, as illustrated in Figure 2.

First, a memory bank is built by extracting the hidden states of stored sentences \(\mathbf{S} = \{s_0, s_1, ..., s_
{S} \, | \, s_i = (t_1, t_2, ..., t_{T_i}) \}\) after \(L_{\mathrm{pre}}\)
Transformer layers. Rather than applying sequence pooling, the complete token-level representations is preserved:

\[ m_j=H^{(L_{\mathrm{pre}})}(s_j), \qquad m_j \in\mathbb{R}^{T_j\times d} \]

Since the sequences have variable lengths, the resulting hidden states are zero-padded to the maximum sequence length,
yielding a memory tensor \(\mathbf{M}\in\mathbb{R}^{S\times T_{\max}\times d}\). A corresponding binary attention mask
identifies valid token positions, ensuring that padded representations are excluded from the late-interaction similarity
computation.

Given an input query, we can extract its hidden states from the \(L_{\mathrm{pre}}\) layers, obtaining
\(Q\in\mathbb{R}^{T_q\times d}\).
The proposed associative retrieval mechanism then iteratively updates this representations through two complementary
sparse weighting operations: sentence-level memory selection and token-level alignment.

First, a late-interaction similarity is computed between each query token and every valid token of each stored memory.
Using cosine similarity, the token-level similarity tensor at iteration \(k\) is defined as:
\[ S_{j,i,t}^{(k)} = \operatorname{cos}\left(Q_i^{(k)},M_{j,t}\right). \]

Following the MaxSim late-interaction formulation, a relevance score is computed for each stored memory by selecting
the maximum token similarity for each query token and averaging the resulting scores:

\[ s_j^{(k)}= \frac{1}{T_q} \sum_{i=1}^{T_q} \max_{t\in\{1,\ldots,T_j\}} S_{j,i,t}^{(k)}. \]

These relevance scores are
transformed into sparse memory-selection weights using an HFY-compatible entmax mapping:

\[ \mathbf{w}^{(k)}= \operatorname{entmax}_{\alpha_{\mathrm{sent}}} (\beta_{\mathrm{sent}}\mathbf{s}^{(k)}). \]

Here, \(\alpha_{\mathrm{sent}}\) controls the sparsity of sentence-level selection, while \(\beta_{\mathrm{sent}}>0\)
is an inverse-temperature parameter that scales the similarity scores and influences the concentration of the resulting
distribution.

In parallel with sentence-level memory selection, a second HFY-compatible entmax operation computes sparse alignments
between each query token and the tokens within each stored memory:
\[ A_{j,i,:}^{(k)} = \operatorname{entmax}_{\alpha_{\mathrm{tok}}} \left(\beta_{\mathrm{tok}} S_{j,i,:}^{(k)} \right).\]

In this case, \(\alpha_{\mathrm{tok}}\) controls the sparsity of token-level alignment, while
\(\beta_{\mathrm{tok}}>0\) determines the concentration of the alignment distribution. Unlike sentence-level selection,
which determines the contribution of each stored sequence, token-level alignment identifies the most relevant token
representations within each memory, preserving fine-grained semantic correspondences.

Finally, both weighting mechanisms are combined to reconstruct an updated query representation as a weighted aggregation
of the original, unnormalized memory hidden states:

\[  Q_i^{(k+1)} = \sum_{j=1}^{N} w_j^{(k)} \sum_{t=1}^{T_j} A_{j,i,t}^{(k)}M_{j,t} \]

where \(w_j^{(k)}\) denotes the \(\alpha\)-entmax weight assigned to memory \(j\) based on its late-interaction
similarity to the current query, and \(A_{j,i,t}^{(k)}\) represents the token-level \(\alpha\)-entmax alignment between
query token \(i\) and memory token \(t\).

After \(K\) associative iterations, the retrieved representation \(R=Q^{(K)}\in\mathbb{R}^{T_q\times d}\) preserves the
query's sequence length and the model's hidden dimension. Finally, rather than reconstructing the retrieved memories as
text, we can prepend the associated representation directly to the query hidden states:

\[ H=[R;Q], \qquad H \in\mathbb{R}^{2T_q\times d} \]

The concatenated representations are then processed by the remaining \(L_{\mathrm{post}}\) Transformer layers, allowing
the model to incorporate retrieved information directly into its internal computation without converting the memories
back into natural language. The resulting hidden states are projected into vocabulary logits through the final
normalisation and language modelling head, from which the next token is selected using greedy decoding. The current
methodology only retrieves the associative memory once, before autoregressive generation begins. However, it reinjects
that same retrieved memory at every decoding step.


## 3. Results and discussion

An initial evaluation was conducted to investigate the feasibility of the proposed Late Interaction
Hopfield–Fenchel–Young architecture, focusing on three aspects: the quality of generated responses, the influence of the
entropic index \(\alpha\) on memory selection, and the evolution of associative retrieval across iterations.

The experiments used **Qwen3.5-4B** [9], extracting contextualized hidden states from Transformer layer 6 of 32. The
memory
bank contained eight sentences, including three directly relevant memories and five contextual or unrelated sentences.

```python
memories = [
    "Fricandó is a traditional Catalan dish. Their main ingredients are slices of beef and mushrooms. It is served with a riched sauce. It can be served at breakfast or lunch.",
    "An special ingredient that Fricandó has is toasted almonds. It is usually the last ingredient to be put.",
    "Fricandó is a typical 'esmorzar de forquilla', that means that is a catalan brunch dish. However, it is known to be eaten during lunch too.",
    "Esmorzar de forquilla is a similar concept of brunch in Catalonia. Normally stews and meat dishes are served.",
    "Sunny and warm weather make people happier.",
    "Rainy and cold weather can negatively affect people's mood.",
    "People tend to spend more time outdoors when the weather is pleasant.",
    'Paella is a traditional Valencian dish. The original version is called "Paella Valenciana".',
]
```

The evaluation query was:

> What are the main ingredients of Fricandó? Where is it from?

Three memories contained relevant information about Fricandó, a traditional catalan dish prepared primarily with beef
and mushrooms, while the remaining memories provided contextual or unrelated information.

The proposed HFY configuration used \(\alpha=1.85\), \(\beta=5\), and four associative iterations. Both sentence-level
selection and token-level alignment used the same \(\alpha\) and \(\beta\) values. The resulting retrieved
representation contained 16 hidden-state vectors of dimension 2560, which were prepended to the query representation at
the selected intermediate layer.

### 3.1. Effect of the convex regularization function \(\Omega\) on associative memory selection

The first experiment investigated how the entropic index \(\alpha\) influences the distribution of retrieval weights.
Five configurations were compared: softmax (\(\alpha=1\)), entmax (\(\alpha=1.5\)), entmax (\(\alpha=1.8\)), sparsemax
(\(\alpha=2\)), and entmax (\(\alpha=2.5\)).

Figure 3 illustrates the resulting distributions across the eight stored memories. The softmax transformation
distributed nonzero weights across all memories, including unrelated information. In
contrast, entmax with \(\alpha=1.8\) assigned zero weight to five memories, concentrating retrieval on the three
directly relevant sentences.

Increasing \(\alpha\) further produced a more concentrated distribution in this example. At \(\alpha=2.5\), the first
two memories accounted for approximately \(95.4\%\) of the total retrieval weight, leaving only \(4.6\%\) for the third
relevant
memory. These results illustrate the influence of \(\alpha\) on retrieval sparsity. While sparse distributions can
reduce contributions from unrelated memories, excessive concentration may also suppress complementary information.

Therefore, the selection of \(\alpha\) introduces a trade-off between memory selectivity and the preservation of
relevant
information distributed across multiple stored sequences.

![alt text](/posts/2026/hfy-late-interaction/sample_weights.png "Fig 3. Effect of the convex regularization function \(\Omega\) on associative memory selection. Different values of \(\alpha\) determine the sparsity of the retrieval distribution: \(\alpha\) = 1 corresponds to softmax, \(\alpha\) = 2 to sparsemax, and intermediate values to \(\alpha\)-entmax. Softmax assigns nonzero weights to all stored memories, whereas \(\alpha\)-entmax can suppress irrelevant memories by assigning them exactly zero weight. Higher \(\alpha\) values produce more concentrated memory-selection distributions in this experiment, reducing interference from unrelated memories")

### 3.2. Generation Quality Across Memory Integration Strategies

The second evaluation compared nine configurations to investigate whether direct memory injection could improve the
factual correctness of generated answers. These included the baseline model, the proposed HFY-based retrieval mechanism,
alternative sparse weighting transformations, and several control conditions.
The configurations and the language model outputs are the following :

- **Baseline** evaluates the LLM's generation capabilities without external memory or additional hidden-state tokens.

```
**Fricandó** is a classic Spanish dish that holds a special place in the culinary history of the Iberian Peninsula. Here are the details regarding its origin and key ingredients:

### Origin
Fricandó is a traditional dish from **Spain**, specifically associated with the region of **Castile and León** (and widely enjoyed throughout the country). It is considered a staple of the *cocina castellana* (Castilian cuisine). The dish has a long history, with roots dating back to the Middle Ages, originally prepared by knights and nobility before becoming a beloved home-cooked meal for commoners. It is often served as a main course or a hearty side dish.

### Main Ingredients
While recipes can vary slightly by family or region, the authentic fricandó is defined by its simplicity and the specific combination of meats and vegetables. The core components are:

*   **Meats**: The dish traditionally features a mix of **pork** and **beef**. Common cuts include pork shoulder (*lomo*), pork belly, and beef shank or chuck. Some variations may include lamb or game meats, but the pork-beef combination is the most standard.
*   **Vegetables**: The primary vegetable is **onion**, which is
```

Baseline output shows that Qwen3.5-4B does not have implicitly the information of the main ingredients of and the region
of where Fricandó is cooked.

- **Reserved-token control** assesses whether prepending additional token positions affects generation independently of
  retrieved memory content.

```
**Fricandó** is a classic Spanish dish that has become a staple in many households, particularly in the Basque Country and the northern regions of Spain.

### Origin
The dish originates from **Spain**, specifically the **Basque Country** (Euskadi). While it is deeply rooted in Basque culinary tradition, it has become so popular across the entire country that it is often considered a national dish. The name itself is derived from the Basque word *fricaz*, meaning "to fry" or "to sauté," which describes the cooking method used.

### Main Ingredients
The authenticity of a traditional fricandó relies on a specific combination of ingredients, though recipes can vary slightly by region. The core components are:

*   **Meat**: The most common cut is **pork loin** (often called *lomo*), but it can also be made with beef, lamb, or chicken. The meat is usually cut into small, bite-sized cubes.
*   **Onions**: A large quantity of **yellow onions** is essential. They are sliced into thin rings or half-moons and form the base of the dish.
*   **Tomatoes**: Fresh **tomatoes** (often peeled and chopped) are added to
```

Introducing additional token positions without meaningful retrieved information change output answer but it does not
improve the geographical attribution.

- **HFY weighted token memory** evaluates sparse associative retrieval using late interaction and \(\alpha\)-entmax
  weighting at
  both sentence and token levels. It has been used \(\alpha = 1.8\).

```
**Fricandó** is a classic Spanish dish that translates to "fried piece" or "braised cutlet." It is a staple of traditional Spanish cuisine, particularly associated with the **Catalonia** region, though it is widely enjoyed throughout Spain.

Here are the details regarding its origin and main ingredients:

### Origin
Fricandó originates from **Catalonia** in northeastern Spain. It is a traditional dish often served as a main course or a tapa. While the specific recipe has evolved over centuries, its roots lie in the rustic cooking traditions of the Catalan countryside, where meat was slow-cooked with local vegetables and herbs to create a rich, flavorful stew.

### Main Ingredients
The dish is defined by its simplicity and the quality of its components. The core ingredients typically include:

*   **Meat**: The most common cut is **pork loin** (often a thick slice or a small roast), though beef or veal can also be used. The meat is usually pounded thin and fried until golden brown before being braised.
*   **Vegetables**: The signature vegetable is **potatoes**, which are sliced into rounds or wedges and added to the pot. Other common additions include:
*   **Onions
```

The proposed sparse associative retrieval mechanism correctly identifies Catalonia as the origin of Fricandó.
Interestingly, the response also mentions ingredients such as pork and potatoes, which are not explicitly included in
the retrieved memories but may appear in alternative preparations. This suggests that the model potentially combines
semantic information transferred through the retrieved hidden states with its existing parametric knowledge, rather than
simply reconstructing the original memory content.

- **Softmax token memory** evaluates dense associative retrieval and the influence of contributions from all stored
  memories.

```
**Fricandó** is a classic Spanish dish that holds a special place in the culinary history of the Iberian Peninsula. Here are the details regarding its origin and main ingredients:

### Origin
Fricandó is a traditional dish from **Spain**, specifically originating in the **Basque Country** (Euskadi). It is a staple of Basque cuisine and is often served as a main course or a hearty side dish. The dish is named after the Basque word *fricaz*, which refers to the method of slow-cooking meat in a sauce.

### Main Ingredients
While recipes can vary slightly by region, the authentic Fricandó is characterized by a specific combination of ingredients that create a rich, savory, and slightly sweet sauce. The core components include:

*   **Meat**: Traditionally made with **pork** (often a mix of pork shoulder and ribs) or **beef**. In some variations, chicken or lamb may be used, but pork is the most common.
*   **Onions**: A large quantity of onions is essential, usually chopped finely and cooked until soft and translucent.
*   **Tomatoes**: Fresh tomatoes (or tomato paste) are added to create the base of the sauce, giving it
```

Dense associative retrieval produces a response associating Fricandó with the Basque Country. Since softmax assigns
nonzero weights to every stored memory, including unrelated sentences, the result is compatible with interference
arising from dense memory aggregation. However, higher steps could let better associative sequence response from HFY
network.

- **Sparsemax token memory** evaluates sparse associative retrieval with \(\alpha\) = 2 and compare it against other
  memory-weighting mechanisms.

```
**Fricandó** is a classic Spanish dish that translates to "fried piece" or "braised cutlet." It is a staple of traditional Spanish cuisine, particularly associated with the **Catalonia** region, though it is widely enjoyed throughout Spain.

Here are the details regarding its origin and main ingredients:

### Origin
Fricandó originates from **Catalonia**, Spain. It is a traditional dish often served as a main course or a tapa. While the specific recipe has evolved over centuries, its roots lie in the rustic cooking traditions of the Catalan countryside, where meat was slow-cooked with local vegetables and herbs to create a rich, flavorful stew.

### Main Ingredients
The dish is defined by its simplicity and the quality of its core components. The traditional recipe typically includes:

*   **Meat**: The most common cut is **pork loin** (often a thick slice or a small roast), though beef or veal can also be used. The meat is usually pounded thin and fried until golden brown before being braised.
*   **Vegetables**: The signature vegetable is **potato**, which is sliced into rounds and fried separately. Other vegetables often added include **carrots**, **onions**, and sometimes **peas
```

Similar to the HFY weighted-token configuration, sparsemax correctly identifies Catalonia while introducing additional
ingredients not explicitly present in the stored memories. This suggests that sparse memory selection may provide
sufficient semantic information to guide geographical identification while allowing subsequent layers to incorporate
additional information from the model's pretrained representations. In this case, pork is considered to be the most
common cut although not being explicitly stated in the context.

- **Random memory control** assesses the effect of injecting norm-matched random hidden states without meaningful
  retrieved information.

```
**Fricandó** is a classic Spanish dish that holds a special place in the culinary history of the Basque Country.

### Origins
The dish is **from the Basque Country** (specifically the region of *Euskal Herria*). It is considered one of the most representative and traditional dishes of Basque cuisine. While it has been enjoyed across Spain for centuries, its roots are deeply tied to the Basque identity and the agricultural traditions of the northern Pyrenees.

### Main Ingredients
The preparation of fricandó is simple, relying on high-quality local ingredients. The main components are:

*   **Beef**: Traditionally, the meat is **beef shank** (the *pierna* of the cow). This cut is chosen because it is rich in connective tissue and fat, which breaks down during the long cooking process to create a tender, flavorful result.
*   **Onions**: Large quantities of **onions** are essential. They are sliced and cooked until they become very soft and sweet, forming the base of the sauce.
*   **Tomatoes**: Fresh **tomatoes** (often peeled and chopped) are added to create a reddish, slightly acidic sauce that balances the richness of the meat
```

Injecting norm-matched random hidden states produces a response associating Fricandó with the incorrect region
from Spain. The result of this configuration is coherent with the baseline, since the latter did not success either.

- **Zero memory control** isolates the effect of prepending zero-valued hidden states without semantic information.

```
**Fricandó** is a classic Spanish dish that holds a special place in the culinary history of the Basque Country.

### Origins
The dish is **from the Basque Country** (specifically the region of *Euskal Herria*). It is considered one of the most representative and traditional dishes of Basque cuisine. While it has been enjoyed across Spain for centuries, its roots are deeply tied to the Basque identity and the agricultural traditions of the northern Pyrenees.

### Main Ingredients
The defining characteristic of fricandó is its simplicity and the use of fresh, local produce. The main ingredients are:

*   **Beef**: Traditionally, the meat is **beef tongue** (*lengua de res*), though in some variations, beef shanks or other cuts of beef are used. The tongue is usually cleaned and cut into small cubes.
*   **Onions**: A large quantity of **onions** is essential. They are sliced and fried until they become soft and translucent, forming the base of the sauce.
*   **Tomatoes**: Fresh **tomatoes** (often peeled and chopped) are added to create a rich, red sauce.
*   **Garlic**: Minced **garlic**
```

Prepending zero-valued hidden states also results in an incorrect geographical attribution. Together with the random
memory control, this suggests that the improvements observed with relevant memory representations cannot be explained
solely by introducing additional hidden-state positions.

- **Top-1 memory injection** evaluates direct injection of the highest-ranked stored memory without weighted aggregation
  across memories.

```
There appears to be a slight confusion in the premise of your question. **Fricandó** is indeed a traditional Catalan dish, but it is **not** made with mushrooms, nor is it typically served at breakfast.

Here are the correct details regarding the dish:

### Origin
**Fricandó** is a classic dish from **Catalonia**, Spain. It is a staple of Catalan cuisine and is often associated with the region's hearty, rustic cooking traditions.

### Main Ingredients
The traditional recipe relies on a specific combination of meats and vegetables, rather than mushrooms:
*   **Meat**: It is traditionally made with **beef** (often a mix of beef and pork, or just beef), cut into small cubes.
*   **Vegetables**: The primary vegetable is **onions**, which are sautéed until soft. **Potatoes** are also a standard component, added to the stew.
*   **Seasoning**: It is seasoned with garlic, paprika, and sometimes a touch of tomato paste or wine.
*   **Sauce**: The dish is cooked slowly in its own juices to create a thick, rich sauce, though it is not usually described as having a "riched sauce" in the sense
```

Direct injection of the highest-ranked stored memory correctly identifies Catalonia and mentions beef. However, the
response explicitly rejects mushrooms as an ingredient, despite their presence in the injected memory. This demonstrates
that the subsequent Transformer layers do not necessarily reproduce retrieved information faithfully and may instead
combine it with existing model knowledge, sometimes producing contradictions.

- **Text retrieval** establishes a conventional text-based RAG baseline by providing retrieved information as
  natural-language context.

```
Based on the provided context, the main ingredients of Fricandó are **slices of beef and mushrooms**. It is a traditional dish from **Catalonia** (Catalan).
```

Conventional text-based RAG correctly identifies Catalonia and the ingredients explicitly mentioned in the retrieved
context: beef and mushrooms. The response is closely aligned with the supplied information, suggesting that textual
context provides a more direct mechanism for grounding generation in the retrieved memories in this initial evaluation.

{{< table caption="Table 1: Qualitative comparison of answer correctness and groundedness across nine generation configurations. The evaluation assesses whether each approach produces factually accurate responses and incorporates information supported by the retrieved memories. Ratings are based on a single preliminary evaluation." >}}

| Configuration             | Answer correctness | Groundedness |
|---------------------------|--------------------|--------------|
| Baseline                  | No                 | -            |
| Reserved-token control    | No                 | -            |
| HFY weighted token memory | Yes                | Partial      |
| Softmax token memory      | No                 | Low          |
| Sparsemax token memory    | Yes                | Partial      |
| Random memory control     | No                 | -            |
| Zero memory control       | No                 | -            |
| Top-1 memory injection    | Low                | Low          |
| Text retrieval            | Yes                | High         |

{{< /table >}}

The initial evaluation suggests that **contextualized hidden-state representations** extracted from intermediate
Transformer
layers **can transfer semantically relevant information to subsequent layers**, even when stored memories and queries
are
encoded independently. In particular, HFY weighted-token memory, sparsemax, and top-1 memory injection correctly
identify Catalonia as the origin of Fricandó, whereas the baseline and control configurations do not.

An interesting observation is that the **generated responses do not simply reproduce the contents of the retrieved
memories**. The HFY and sparsemax configurations introduce additional ingredients that are not explicitly represented in
the memory bank but may correspond to alternative preparations of the dish. This behavior is consistent with the
hypothesis that the remaining Transformer layers integrate information from the injected memory representations with
knowledge encoded in the pretrained model parameters, rather than treating retrieved memories as an exclusive source of
information.

The architecture is particularly interesting because memory sentences and the query are initially processed
independently through the first \(L_{\mathrm{pre}}\) Transformer layers. No cross-sequence attention is computed between
them during this initial encoding. Nevertheless, after the retrieved representations are prepended to the query hidden
states, the remaining \(L_{\mathrm{post}}\) layers can establish contextual interactions between both sequences and use
this information during generation. Investigating iterative associative memory retrieval could be an interesting
research direction to be considered, as partially generated responses could provide additional context to improve
memory retrieval and memory-query-answer cross-attention.

However, top-1 memory injection shows how a raw hidden state with the information cannot let the model answer correctly.
This demonstrates that is the **result of query and associated memories** what gives **enough information** to the
model (specifically, to the remaining \(L_{\mathrm{post}}\) layers) to answer with success.

Compared with direct hidden-state injection, **conventional text retrieval** produces a response **more closely grounded
in the retrieved context**. Nevertheless, this does not necessarily imply that its generation mechanism is preferable in
every setting, since text-based retrieval and hidden-state injection provide information through different
representations and may encourage different degrees of reliance on retrieved versus parametric knowledge. Since the
language model has been trained on instruction-following and question answering datasets [9], they are generally better
adapted to incorporating information provided as natural-language context than as intermediate hidden-state
representations. This finding can induce to create **joint
post-training strategies** that explicitly train language models to interpret and integrate retrieved associative
memories. Given the differentiable nature of HFY networks, such approaches could enable end-to-end optimizaition of
memory
selection and downstream generation, potentially improving the model's ability to exploit associative memories directly
within its hidden-state space.

### 3.3. Query Similarity Across Associative Iterations

The final experiment examined the evolution of memory similarity scores and retrieval weights over four associative
iterations.

Figure 4 presents the late-interaction similarity scores and corresponding entmax selection weights for the initial
query \(Q^{(0)}\) and the updated states \(Q^{(1)},\ldots,Q^{(4)}\).

![alt text](/posts/2026/hfy-late-interaction/query_similarity.png "Fig 4. Evolution of memory similarity and sparse associative selection across four HFY retrieval iterations. The left heatmap shows the MaxSim similarity between the evolving query representations \(Q^{(0)},\ldots,Q^{(4)}\) and the eight stored memories. The right heatmap illustrates the corresponding \(\alpha\)-entmax selection weights.")

The results show a **substantial increase** in similarity **after the first associative update**, particularly for the
three
most relevant memories. Subsequent iterations produced smaller changes, suggesting progressive stabilisation of the
retrieved representations. Self-reinforcing subsequent iterations shall be furtherly explored, since subsequent
iterations tend to favor those same memories.

Despite the increase in similarity scores for several stored memories, the entmax transformation consistently assigned
zero weight to the five unselected memories. This indicates that associative updates can modify similarity relationships
while maintaining a sparse memory-selection distribution.

## 4. Conclusion

This work explored a **Late Interaction Hopfield-Fenchel-Young architecture** for **sparse associative retrieval** and
direct integration of **token-level memories into large language models**.

The preliminary experiments with Qwen3.5-4B provide initial evidence that **retrieved intermediate hidden states can
influence downstream generation** and **transfer semantically relevant information**, even when no cross-atention
between queries and stored memories are calculated inside model. For instance, new flash models such as DeepSeek V4.1
Flash, that uses encoder and decoder architucture, could be benefitted from this system.

Nevertheless, the results also reveal an important limitation: **successful semantic transfer does not guarantee
fully-grounded generation**. Although sparse associative retrieval improved certain factual aspects of the generated
responses, the model frequently introduced information absent from the retrieved memories. In contrast, conventional
text-based retrieval produced a more accurate and context-grounded response. In addition, the current qualitative
evaluation is insufficient to establish general improvements in answer correctness, retrieval efficiency, or robustness.

Several architectural and methodological challenges therefore remain open. First, the current retrieval mechanism
constructs an associated representation with the same sequence length as the query, potentially limiting the
preservation of information from longer stored memories. Second, no experiments were carried out when no grounded
evidence is found in the memory. Third, the current implementation aggregates retrieved information into a query-aligned
representation, producing one memory vector per query token. Consequently, information from longer memories must be
compressed into the query's sequence length, potentially limiting information preservation. Addressing these limitations
will require evaluating alternative memory-preservation strategies and include guardrails when retrieving.

A particularly promising research direction is the development of **joint post-training strategies for associative
retrieval and language generation**. Since HFY retrieval transformations can be differentiable, they provide a potential
foundation for optimizing memory selection and downstream generation together.

Overall, these findings indicate that **sparse token-level associative memories offer a viable experimental mechanism
for transferring information through continuous hidden-state representations**. Bridging this gap represents an
important direction for developing language models capable of integrating external knowledge directly within their
internal representation space.

## References

[1]    H. Ramsauer et al., ‘Hopfield Networks is All You Need’, Apr. 28, 2021, arXiv: arXiv:2008.02217. doi:
10.48550/arXiv.2008.02217.

[2]    S. J. R. D. Santos, V. Niculae, D. C. Mcnamee, and A. Martins, ‘Sparse and Structured Hopfield Networks’, in
Proceedings of the 41st International Conference on Machine Learning, PMLR, Jul. 2024, pp. 43368–43388. [Online].
Available: https://proceedings.mlr.press/v235/santos24a.html

[3]    S. Santos, V. Niculae, D. McNamee, and A. F. T. Martins, ‘Hopfield-Fenchel-Young Networks: A Unified Framework
for Associative Memory Retrieval’, Oct. 26, 2025, arXiv: arXiv:2411.08590. doi: 10.48550/arXiv.2411.08590.

[4]    O. Khattab and M. Zaharia, ‘ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction
over BERT’, Jun. 04, 2020, arXiv: arXiv:2004.12832. doi: 10.48550/arXiv.2004.12832.

[5]    X. Cheng et al., ‘xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token’, in
Advances in Neural Information Processing Systems, Curran Associates, Inc., 2024, pp. 109487–109516. doi:
10.52202/079017-3476.

[6]    E. Spens and N. Burgess, ‘Hippocampo-neocortical interaction as compressive retrieval-augmented generation’, Nat.
Commun., vol. 17, no. 1, p. 7971, Jun. 2026, doi: 10.1038/s41467-026-74357-6.

[7]    A. Jain, X. Yao, T. Reps, and C. Jermaine, ‘MHA-RAG: Improving Efficiency, Accuracy, and Consistency by Encoding
Exemplars as Soft Prompts’, arXiv.org. Accessed: Nov. 11, 2026. [Online]. Available: https://arxiv.org/abs/2510.05363v2

[8]    R. Wei et al., ‘Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models’, arXiv.org.
Accessed: Nov. 11, 2026. [Online]. Available: https://arxiv.org/abs/2610.25100v1

[9] Q. Team, ‘Qwen3.5-Omni Technical Report’, Apr. 21, 2026, arXiv: arXiv:2604.15804. doi: 10.48550/arXiv.2604.15804.

## Citation

```
@article{alas2026,
  title   = "Late Interaction Hopfield-Fenchel-Young Network for Sparse Retrieval of Language Model Hidden States",
  author  = "Alàs Cercós, Oriol",
  journal = "oriolac.github.io",
  year    = "2026",
  month   = "November",
  url     = "https://oriolac.github.io/posts/20261111-hfy-late-interaction/"
}
```

