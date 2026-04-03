# Proper Usage 

## Reranker

Model Architecture:

Architecture Type: Transformer
Network Architecture: Fine-tuned meta-llama/Llama-3.2-1B
Max Sequence Length: 8192
Number of Model Parameters: 1.0 × 10^9

This reranking model is a transformer encoder fine-tuned for ranking. Ranking models for text retrieval are typically trained as a cross-encoder for sentence classification, predicting the relevancy of a sentence pair (for example, question and chunked passages). Cross-entropy loss is used to maximize the likelihood of passages containing information to answer the question and minimize the likelihood for negative passages that do not.

Input:

Input Type: Pair of texts (query + passage)
Input Format: List of text pairs / JSON payload (query + passages)
Input Parameters: One Dimensional (1D)
Other Input Properties: Evaluated to work successfully with up to a sequence length of 8192 tokens. Longer texts should be chunked or truncated.

Output:

Output Type: Floats (logits / scores)
Output Format: List of floats (scores per passage)
Output Parameters: One Dimensional (1D)
Other Output Properties: Users may apply a sigmoid activation function to logits if desired.


## Embedder

Model Architecture:

Architecture Type: Transformer encoder
Network Architecture: Fine-tuned Llama 3.2 1B retriever
Embedding Dimension: 2048
Max Sequence Length: 8192
Number of Model Parameters: 1B

This embedding model is a transformer encoder fine-tuned for contrastive learning using a bi-encoder setup. Query and passage texts are encoded independently, and contrastive learning maximizes similarity for relevant query–passage pairs while minimizing similarity for negative passages. The model supports Matryoshka embeddings to enable dynamic output dimensions.

Input(s):

Input Type(s): Text
Input Format(s): String / List of strings
Input Parameters: One Dimensional (1D)
Other Properties Related to Input: Text inputs longer than the maximum context length must be truncated or chunked.

Output(s)

Output Type(s): Floats (dense vector embeddings)
Output Format(s): List of floats
Output Parameters: One Dimensional (1D)
Other Properties Related to Output: Embedding vectors can be configured to output dimensions of 384, 512, 768, 1024, or 2048.


## Temporal Decay upgrades

The Decay theory is a theory that proposes that memory fades due to the mere passage of time. Information is therefore less available for later retrieval as time passes and memory, as well as memory strength, wears away.[1] When an individual learns something new, a neurochemical "memory trace" is created. However, over time this trace slowly disintegrates. Actively rehearsing information is believed to be a major factor counteracting this temporal decline.[2] It is widely believed that neurons die off gradually as we age, yet some older memories can be stronger than most recent memories. Thus, decay theory mostly affects the short-term memory system, meaning that older memories (in long-term memory) are often more resistant to shocks or physical attacks on the brain. It is also thought that the passage of time alone cannot cause forgetting, and that decay theory must also take into account some processes that occur as more time passes.[1]f
> In similar fashion, temporal decay should affect the temporary aspects of memory storage, the transient, the memories
> Switch to inference theory
