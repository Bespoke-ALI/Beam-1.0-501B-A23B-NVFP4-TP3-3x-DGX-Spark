# SPDX-License-Identifier: Apache-2.0
# Serving image for Beam-1.0 NVFP4 at TP=3: the pinned upstream vLLM image plus one changed constant.
# vLLM pads the vocabulary to a multiple of DEFAULT_VOCAB_PADDING_SIZE before splitting it across ranks.
# 151,936 is already a multiple of 64, so vLLM leaves it unpadded, and it does not divide by 3.
# Padding to a multiple of 192 (3 x 64) gives 152,064 = 3 x 50,688; vLLM masks the 128 padded logit rows.
# The grep -qx checks make the build fail if the upstream line ever changes, instead of silently not patching.
# NOTICE (Apache-2.0 section 4(b)): this build modifies vllm/model_executor/layers/vocab_parallel_embedding.py.
FROM vllm/vllm-openai@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77
RUN set -e; \
    F="$(python3 -c 'import importlib.util as u; print(u.find_spec("vllm").submodule_search_locations[0])')/model_executor/layers/vocab_parallel_embedding.py"; \
    grep -qx 'DEFAULT_VOCAB_PADDING_SIZE = 64' "$F"; \
    sed -i 's/^DEFAULT_VOCAB_PADDING_SIZE = 64$/DEFAULT_VOCAB_PADDING_SIZE = 192/' "$F"; \
    grep -qx 'DEFAULT_VOCAB_PADDING_SIZE = 192' "$F"
LABEL org.opencontainers.image.description="vLLM with DEFAULT_VOCAB_PADDING_SIZE 64->192 so Beam's 151,936-token vocabulary splits evenly at TP=3" \
      org.opencontainers.image.base.digest="sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77" \
      org.opencontainers.image.licenses="Apache-2.0"
