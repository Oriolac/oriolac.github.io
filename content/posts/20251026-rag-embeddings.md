+++
title = 'Embedding models'
date = 2025-02-17T12:31:23+01:00
draft = true
tags = ['nlp', 'introduction', 'transformers', 'embedding-models', 'deep-learning']
metaDescription = ''
[cover]
image = "/posts/2024/att-post.PNG"
+++

# Embeddings in Retrieval-Augmented Generation (RAG) pipelines

**Retrieval-Augmented Generation (RAG)** is a design pattern for question answering and other NLP tasks where a
generative
model is augmented with a information retrieval step. The idea is straightforward: instead of relying
solely on what’s in the model’s parameters, we give the model access to an external knowledge base. When a user asks a
question, the system first retrieves relevant text from a document corpus, and then the generative model conditions its
answer on that text. This approach can significantly improve the factual accuracy of the answer and reduce
hallucinations because the language model has specific, relevant data to draw from.

Where do embeddings come into play? In the **retriever**. A typical RAG pipeline uses a vector database or index of
embeddings for all the documents (or passages) in the knowledge base. When a query comes in, the pipeline uses an
embedding model to encode the query into a vector. Then it searches for the most similar vectors (via nearest neighbor
search) among the stored document embeddings. Those top-ranked documents are fetched as context for the generative
model, which then reads them (along with the question) and composes the final answer.

**The quality of the embedding model in the retriever is critical.** If the embeddings are poor (they don’t
adequately capture the meaning of queries or documents) the retriever might return irrelevant texts. The generator then
has the wrong context and will produce a wrong answer (garbage in, garbage out). Conversely, a high-quality embedding
model tuned for your type of questions and documents will retrieve very relevant information, making the generator’s job
easy (and keeping it grounded in truth).

RAG pipelines can use different types of retrievers. Some systems use dense retrievers exclusively (embedding-based
search), while others use or add sparse retrieval (keyword search) or even structured retrieval (SQL or graph queries)
depending on the data. But increasingly, embedding-based retrieval is popular because of its ability to handle natural
language queries and vocabulary mismatches gracefully. In practice, many implementations use a hybrid: e.g., first use
an embedding model to get candidate passages by semantic similarity, and perhaps re-rank or filter them with a
keyword-based method to ensure exact requirements are met.

To sum up, in a RAG system for Q&A, the embedding model is the brains of the retrieval component. It translates text
into the vector language of meaning, enabling the system to fetch knowledge by content. A good embedding model ensures
that when the user asks something, the system finds the right information (e.g. the relevant paragraph in the
documentation) to feed to the language model. As we’ll see, fine-tuning embedding models can make this process even more
accurate by tailoring the notion of “similarity” to what’s truly relevant in your application.

# Fine-Tuning Embeddings for Domain QA

## Training objectives for Embedding Model Fine-Tuning