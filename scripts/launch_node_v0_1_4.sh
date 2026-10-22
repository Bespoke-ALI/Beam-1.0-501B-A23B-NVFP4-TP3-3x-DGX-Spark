#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# launch_node_v0_1_4.sh - start one rank of Beam-1.0 NVFP4 at TP=3. Run the same script on all three nodes.
# Usage:   NODE_RANK=0 MASTER_ADDR=<node0 LAN IP> bash launch_node_v0_1_4.sh      (node0 serves the API)
#          NODE_RANK=1 MASTER_ADDR=<node0 LAN IP> bash launch_node_v0_1_4.sh      (headless worker)
#          NODE_RANK=2 MASTER_ADDR=<node0 LAN IP> bash launch_node_v0_1_4.sh      (headless worker)
# Every check below must pass or nothing starts. Stop the containers with: docker rm -f beam-tp3
set -u
NODE_RANK="${NODE_RANK:?set NODE_RANK to 0, 1 or 2}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the LAN address of node0}"
LAN_IF="${LAN_IF:-enP7s7}"                                     # the Spark's built-in Ethernet port
IMG="${IMG:-beam-tp3:v1}"                                      # built from docker/Dockerfile
MODEL="${MODEL:-/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3}" # the padded checkpoint
PORT="${PORT:-8600}"
MIN_FREE_GIB="${MIN_FREE_GIB:-109}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"                        # lower it if vLLM reports too little KV cache memory; see README
GPU_UTIL="${GPU_UTIL:-0.885}"                  # 0.88 left too little KV-cache memory on some launches; see README
MAX_BATCHED="${MAX_BATCHED:-}"           # optional: --max-num-batched-tokens (smaller = less start-up memory, slower prefill)
KV_BYTES="${KV_BYTES:-}"                 # optional: --kv-cache-memory-bytes (fixes the KV cache size; skips vLLM's estimate)
LOG_LEVEL="${LOG_LEVEL:-INFO}"           # DEBUG prints vLLM's full start-up memory breakdown
RING_PORTS="enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1"
OK=1; fail() { echo "STOP: $*"; OK=0; }

case "$NODE_RANK" in 0|1|2) ;; *) fail "NODE_RANK must be 0, 1 or 2";; esac
IP=$(ip -4 -o addr show "$LAN_IF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1)
echo "NODE $(hostname) rank=$NODE_RANK lan_ip=${IP:-none} master=$MASTER_ADDR max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_UTIL max_batched=${MAX_BATCHED:-default} kv_bytes=${KV_BYTES:-auto}"
[ -n "$IP" ] || fail "no IPv4 address on $LAN_IF (set LAN_IF)"
[ "$NODE_RANK" = 0 ] && [ "$IP" != "$MASTER_ADDR" ] && fail "rank 0 must run on MASTER_ADDR ($MASTER_ADDR); this node is $IP"
docker image inspect "$IMG" >/dev/null 2>&1 || fail "image $IMG not found (build docker/Dockerfile)"
[ "$( (wc -l < "$MODEL/SHA256SUMS") 2>/dev/null)" = 64 ] || fail "padded checkpoint missing at $MODEL (want 64 lines in SHA256SUMS)"
docker ps -a --format '{{.Names}}' | grep -qx beam-tp3 && fail "a beam-tp3 container already exists (docker rm -f beam-tp3)"
for p in $RING_PORTS; do [ "$(cat /sys/class/net/$p/operstate 2>/dev/null)" = up ] || fail "ring port $p is not up"; done
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
GP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
[ "$GP" = 0 ] || fail "$GP process(es) already use the GPU; stop every other GPU workload first"
if [ "$OK" = 1 ]; then
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 3   # GB10 memory is shared; free the page cache
  AV=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  echo "MEM_AVAILABLE_GIB=$AV (need >= $MIN_FREE_GIB)"
  [ "$AV" -ge "$MIN_FREE_GIB" ] || fail "not enough free memory"
fi
if [ "$OK" != 1 ]; then echo "NOTHING STARTED"; else
  EXTRA="--headless"
  OPT=""; [ -n "$MAX_BATCHED" ] && OPT="$OPT --max-num-batched-tokens $MAX_BATCHED"; [ -n "$KV_BYTES" ] && OPT="$OPT --kv-cache-memory-bytes $KV_BYTES"
  [ "$NODE_RANK" = 0 ] && EXTRA="--host 127.0.0.1 --port $PORT --served-model-name beam --reasoning-parser beam --tool-call-parser beam --enable-auto-tool-choice"
  docker run -d --name beam-tp3 --network host --ipc host --gpus all --device /dev/infiniband \
    --ulimit memlock=-1 --ulimit stack=67108864 -v "$MODEL:$MODEL:ro" \
    -e VLLM_HOST_IP="$IP" -e GLOO_SOCKET_IFNAME="$LAN_IF" -e NCCL_SOCKET_IFNAME="$LAN_IF" \
    -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_NET_PLUGIN=none -e NCCL_DEBUG=WARN \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_LOGGING_LEVEL="$LOG_LEVEL" \
    "$IMG" "$MODEL" --tensor-parallel-size 3 --nnodes 3 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port 29510 --distributed-executor-backend mp \
    --moe-backend b12x --load-format safetensors --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 --enforce-eager $OPT $EXTRA >/dev/null \
  && sleep 10 && echo "STARTED beam-tp3 rank=$NODE_RANK state=$(docker inspect -f '{{.State.Status}}' beam-tp3)" \
  && echo "Start-up takes about 65-70 minutes, including a long silent phase after the weights load. Do not stop it. Follow it with: docker logs -f beam-tp3" \
  && { # GB10 memory is shared: page cache filled by reading the weights counts against vLLM's KV-cache estimate.
       # After the weights finish loading, drop the page cache every 10 s until vLLM has sized its KV cache.
       sudo -b sh -c 'loaded=0; for i in $(seq 1 720); do [ "$(docker inspect -f "{{.State.Running}}" beam-tp3 2>/dev/null)" = true ] || exit 0;
         docker logs beam-tp3 2>&1 | grep -qE "Available KV cache memory|GPU KV cache size" && exit 0;
         [ $loaded = 1 ] || { docker logs beam-tp3 2>&1 | grep -qE "checkpoint shards: +100%|Model loading took|MoEPrepareAndFinalize" && loaded=1; };
         [ $loaded = 1 ] && { sync; echo 1 > /proc/sys/vm/drop_caches; }; sleep 10; done' >/dev/null 2>&1;
       echo "page-cache helper running until the KV cache is sized"; } \
  || echo "FAILED: docker run did not start beam-tp3 (see the error above)"
fi
