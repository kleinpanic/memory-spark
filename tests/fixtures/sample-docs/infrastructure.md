# Infrastructure Notes

## Network
The primary compute node is reachable at 192.0.2.1 via a WireGuard tunnel.
The gateway binds to loopback and uses reverse tunnels for external access.

## Services
- **vLLM**: Runs large language models on GPU
- **Embedding**: Uses nvidia/llama-embed-nemotron-8b for vector embeddings
- **Reranker**: Uses nvidia/llama-nemotron-rerank-1b-v2 for result reranking

## Restart Protocol
The ONLY approved restart mechanism is `oc-restart`. Direct `systemctl restart` is banned.
Every restart requires explicit approval from the human operator.
