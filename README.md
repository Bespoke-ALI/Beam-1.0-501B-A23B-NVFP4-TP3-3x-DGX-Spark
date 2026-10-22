<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Beam-1.0 501B on three DGX Sparks at TP=3

Run Reflection's [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) (NVFP4) across three NVIDIA DGX Spark (GB10) nodes with tensor parallelism 3, on a stock vLLM image with one constant changed.

Beam has 8 key-value heads, and three other widths that don't divide by 3, so it cannot start at TP=3 as shipped. This repo pads the checkpoint offline with zeros so every split is even, shows analytically (under stated kernel assumptions) that the padded model computes the same function, and verifies the padded files byte by byte against the original. Then it walks through serving it on a switchless three-node ring, with a pass/fail check after every step. The only change to vLLM is one constant, patched into the pinned upstream image.

**Measured on our cluster:**
- about 18 tokens/s single-stream decode;
- a correct tool call (one probe);
- exact retrieval of a planted fact from a 41,522-token prompt;
- 131 of 133 checks passed in Reflection's own [Beam Verifier](https://github.com/reflection-oss/beam-verifier), including every structured-output and tool-calling check (the 2 failures are one request-validation check; see [Results](#results));
- GPQA Diamond and IFBench scores consistent with Reflection's published NVFP4 figures ([Results](#results));
- about 3 hours from starting the download to the first token.

To the best of our knowledge, this is the first public recipe for running Reflection's own NVFP4 (4-bit) weights on DGX Spark, the first Beam setup across several Sparks, and the first at TP=3. That rests on a search of Hugging Face, GitHub, vendor documentation and the web as of Oct 21, 2026, which was not exhaustive. A different approach, a single-Spark build that re-quantizes the experts to about 1.5 bits and runs on its own engine, was published on Oct 20, 2026 (see [Related work](#related-work-and-other-builds)). TP=3 on Sparks itself is established for other models.

> Not affiliated with or endorsed by NVIDIA or Reflection AI. NVIDIA, DGX Spark, GB10 and ConnectX are trademarks of NVIDIA Corporation; Beam is Reflection AI's model. Names are used only to describe what this was run on.

## Contents

1. [Tested on](#tested-on)
2. [What you need](#what-you-need)
3. [Step 0: prepare every node](#step-0-prepare-every-node)
4. [Step 1: ring network](#step-1-ring-network)
5. [Step 2: download and verify the checkpoint](#step-2-download-and-verify-the-checkpoint)
6. [Step 3: pad, verify and test](#step-3-pad-verify-and-test)
7. [Step 4: build the serving image](#step-4-build-the-serving-image)
8. [Step 5: copy to the other nodes](#step-5-copy-to-the-other-nodes)
9. [Step 6: launch](#step-6-launch)
10. [Step 7: smoke tests](#step-7-smoke-tests)
11. [Results](#results)
12. [Why it is built this way](#why-it-is-built-this-way)
13. [Troubleshooting](#troubleshooting)
14. [Limitations](#limitations)
15. [Related work and other builds](#related-work-and-other-builds)
16. [Credits, citation and licence](#credits-citation-and-licence)

## Tested on

All three nodes were identical. Recorded Oct 20, 2026, from the hosts and the running containers; nothing was changed before the last measurement on Oct 21.

| Component | Version |
|---|---|
| Hardware | 3 × DGX Spark (GB10, 128 GB unified memory, ConnectX-7 with two 200 Gb/s QSFP ports) |
| OS | DGX OS 7.5.0, kernel 6.17.0-1026-nvidia, aarch64 |
| NVIDIA driver | 595.71.05 |
| ConnectX-7 firmware | 28.45.4028 |
| Host test tools | NCCL 2.30.7 (`libnccl2 2.30.7-1+cuda13.3`), Open MPI 4.1.6 (`4.1.6-7ubuntu2`), perftest 24.01 (`24.01.0+0.38-1build2`, `ib_write_bw` 6.20), nccl-tests commit `358ca293`, built against them |
| Base image | `vllm/vllm-openai@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77` (arm64, Beam day-0 support build) |
| Inside the image | vLLM 0.1.dev23042+g8e0922e88, PyTorch 2.13.0+cu130, FlashInfer 0.7.2.dev20261014 (includes the b12x MoE kernels), NCCL 2.30.7 (`nvidia-nccl-cu13`), transformers 5.19.0 |
| Model | `reflection/Beam-1.0-501B-A23B-NVFP4`, revision `3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086`, Apache-2.0; 63 files, 293,654,927,078 bytes |

Other driver, kernel or vLLM versions are untested.

## What you need

- **Three DGX Sparks and three QSFP cables** (200 Gb/s QSFP112 direct-attach) for a switchless ring.
- **Docker and the NVIDIA Container Toolkit** on every node; your user in the `docker` group and able to use `sudo` (netplan, dropping the page cache, creating `/srv/models`).
- **Passwordless SSH** from node0 to the other two nodes (for the NCCL test), and from node0 and node2 to the build node (for copying).
- **Disk:**
  - about 650 GB free on the build node (original, padded copy and images);
  - about 360 GB on each other node.
  - Models live under `/srv/models`, owned by your user: `sudo install -d -o "$USER" -g "$USER" /srv/models`.
- **Memory at launch:** at least 109 GiB `MemAvailable` on every node, after dropping the page cache. Stop every other GPU workload first. GB10 memory is shared between CPU and GPU, so anything else resident competes with the weights.
- **A build node with a fast internet link.** We used node1. One of our nodes had negotiated a 100 Mb/s management link; downloading on it would have taken over 6 hours.

Node roles in this guide:

| Node | TP rank | Role |
|---|---|---|
| node0 | 0 | Serves the API; rendezvous master |
| node1 | 1 | Worker; build node (download and padding) |
| node2 | 2 | Worker |

## Step 0: prepare every node

Run every command in this guide from the repo root unless a step says otherwise. Each step re-declares the variables it needs, so a new shell is fine.

On **every** node:

```bash
git clone https://github.com/Bespoke-ALI/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark && cd Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark
sudo apt install -y perftest=24.01.0+0.38-1build2 openmpi-bin=4.1.6-7ubuntu2 libopenmpi-dev=4.1.6-7ubuntu2 \
  libnccl2=2.30.7-1+cuda13.3 libnccl-dev=2.30.7-1+cuda13.3   # the versions we ran; NCCL from NVIDIA's CUDA apt repository
sudo install -d -o "$USER" -g "$USER" /srv/models /opt/nccl-tests
git clone https://github.com/NVIDIA/nccl-tests /opt/nccl-tests && git -C /opt/nccl-tests checkout 358ca293f16b1248fead83353520d5768fee07c8
make -C /opt/nccl-tests MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda
```

- These are the exact package versions and nccl-tests commit we ran (see [Tested on](#tested-on)); other versions are untested. If apt no longer offers a pinned version, install the nearest one and expect to re-check Step 1.
- nccl-tests is needed at the same path on all three nodes, because `mpirun` starts it on each.

**SSH.** From node0, `ssh-copy-id <user>@<node1-address>` and `ssh-copy-id <user>@<node2-address>` (for the NCCL test). From node0 and node2, also copy a key to node1, the build node (for Step 5).

**Pass:** `ls /opt/nccl-tests/build/all_gather_perf` exists on every node, and `ssh -o BatchMode=yes <user>@<address> true` succeeds from node0 to both workers.

## Step 1: ring network

Cable the triangle exactly as NVIDIA's [three-Spark ring playbook](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring) specifies: **every cable joins a port 0 on one node to a port 1 on the next.**
- Port 0 is the QSFP port next to the Ethernet port.
- We first cabled one link port 1 to port 1, and pair tests over it failed until we re-plugged.

| Cable | End A | End B |
|---|---|---|
| C1 | node0 port 1 | node1 port 0 |
| C2 | node1 port 1 | node2 port 0 |
| C3 | node2 port 1 | node0 port 0 |

Each QSFP port appears to Linux as two network interfaces and two RDMA devices, one per PCIe half:
- port 0: `enp1s0f0np0` / `rocep1s0f0` and `enP2p1s0f0np0` / `roceP2p1s0f0`;
- port 1: the same names with `f1np1` / `f1`.

We give each cable two /24 subnets (one per half), jumbo frames, and no DHCP or IPv6 link-local addressing. This addressing is ours. NVIDIA's playbook uses a different scheme; either works as long as both ends of each cable share subnets.

Write `/etc/netplan/40-ring.yaml` on each node (`sudo chmod 600` it), then `sudo netplan apply`.

**node0**
```yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.100.1/24]}
    enP2p1s0f1np1: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.101.1/24]}
    enp1s0f0np0:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.102.1/24]}
    enP2p1s0f0np0: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.103.1/24]}
```

**node1**
```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.100.2/24]}
    enP2p1s0f0np0: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.101.2/24]}
    enp1s0f1np1:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.104.1/24]}
    enP2p1s0f1np1: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.105.1/24]}
```

**node2**
```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.104.2/24]}
    enP2p1s0f0np0: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.105.2/24]}
    enp1s0f1np1:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.102.2/24]}
    enP2p1s0f1np1: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.103.2/24]}
```

**Check every cable: six interface pairs, three cables × two halves.** For each row, start the server, then run the client:
- on the server node: `ib_write_bw -d <server device> -x 3 -F --report_gbits -s 1048576 -D 5`
- on the client node: `ping -M do -s 8972 -c 3 <server address>` and `ib_write_bw -d <client device> -x 3 -F --report_gbits -s 1048576 -D 5 <server address>`

| Cable, half | Server node, device | Server address | Client node, device |
|---|---|---|---|
| C1, 1 | node1 `rocep1s0f0` | 192.168.100.2 | node0 `rocep1s0f1` |
| C1, 2 | node1 `roceP2p1s0f0` | 192.168.101.2 | node0 `roceP2p1s0f1` |
| C2, 1 | node2 `rocep1s0f0` | 192.168.104.2 | node1 `rocep1s0f1` |
| C2, 2 | node2 `roceP2p1s0f0` | 192.168.105.2 | node1 `roceP2p1s0f1` |
| C3, 1 | node2 `rocep1s0f1` | 192.168.102.2 | node0 `rocep1s0f0` |
| C3, 2 | node2 `roceP2p1s0f1` | 192.168.103.2 | node0 `roceP2p1s0f0` |

`-x 3` selects the RoCE v2 IPv4 address entry (GID index 3) on our nodes. Confirm yours and change it if needed: `cat /sys/class/infiniband/<device>/ports/1/gids/3` should show your IPv4 address in IPv6-mapped form, and `cat /sys/class/infiniband/<device>/ports/1/gid_attrs/types/3` should print `RoCE v2`.

**Then a three-node NCCL test from node0.** nccl-tests must be built against MPI. `-H` takes any address each node is reachable on; `enP7s7` is the Spark's built-in Ethernet port, used here only for setup traffic.

```bash
export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_MERGE_NICS=0 NCCL_NET_PLUGIN=none
mpirun -np 3 -H localhost:1,<node1-address>:1,<node2-address>:1 --mca plm_rsh_agent "ssh -o BatchMode=yes" \
  --mca btl_tcp_if_include enP7s7 --mca oob_tcp_if_include enP7s7 -x LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x NCCL_IB_HCA -x NCCL_IB_SUBNET_AWARE_ROUTING -x NCCL_IB_MERGE_NICS -x NCCL_NET_PLUGIN -x NCCL_DEBUG=INFO \
  /opt/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

**Pass:**
- all 12 ring interfaces up at 200,000 Mb/s (`cat /sys/class/net/<if>/speed`);
- jumbo pings with 0% loss;
- every interface pair at about 109 Gb/s (we measured 109.27–109.34);
- the all-gather completes. We measured 23.9 GB/s average bus bandwidth.

| NCCL variable | Value | Why |
|---|---|---|
| `NCCL_IB_HCA` | `rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` | All four RDMA devices are eligible |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | Pick the local device on the peer's subnet |
| `NCCL_IB_MERGE_NICS` | `0` | Keep the two halves of a port as separate NICs |
| `NCCL_NET_PLUGIN` | `none` | Use NCCL's built-in RoCE transport |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | `enP7s7` | Bootstrap over the LAN |

## Step 2: download and verify the checkpoint

**Shortcut: download the padded checkpoint instead (skips Steps 2 and 3).** The output of Step 3 is published at [Bespoke-ALI/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark](https://huggingface.co/Bespoke-ALI/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark). On the build node:

```bash
REPO=Bespoke-ALI/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark
REV=main      # or pin a commit from the repo's history
DST=/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
mkdir -p "$DST"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 -v "$DST:/out" \
  python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 sh -c \
  "pip install -q --target /tmp/pkgs huggingface_hub==1.27.0 && PYTHONPATH=/tmp/pkgs python -c \
  \"from huggingface_hub import snapshot_download as s; s(repo_id='$REPO', revision='$REV', local_dir='/out', max_workers=8)\""
(cd "$DST" && sha256sum SHA256SUMS && sha256sum --quiet -c SHA256SUMS && echo SUMS_MATCH)
python3 scripts/check_sums_v0_1_0.py "$DST"
```

- **Pass:** the first line printed is `9f57e87b492a27c3fa3cea11b0e45e1f4784069a35ee05ea2b3e0f1c100cd5dd  SHA256SUMS`, then `SUMS_MATCH` and `SUMS_VS_MANIFEST=OK`. The fingerprint is published here on GitHub, separately from the Hugging Face repo, so a changed checksum list cannot pass unnoticed.
- `SHA256SUMS` lists the 64 files the padding step produced. Reflection's original model card is listed under its new name, `README_reflection.md`. The Hugging Face model card (`README.md`) and `NOTICE` are not listed.
- If the Hugging Face page still shows an "upload in progress" notice, the files are incomplete; build it yourself with Steps 2 and 3 instead.
- Then continue at Step 4.

**Or build it yourself (Steps 2 and 3).** Run on the build node. The revision is pinned. The download runs in a throwaway container, so no Hugging Face client is installed on the host.

```bash
REPO=reflection/Beam-1.0-501B-A23B-NVFP4
REV=3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086
SRC=/srv/models/Beam-1.0-501B-A23B-NVFP4
W=/srv/models/_manifests
mkdir -p "$SRC" "$W"
curl -fsS "https://huggingface.co/api/models/$REPO" -o "$W/api_model.json"
curl -fsS "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=true" -o "$W/tree_$REV.json"
echo "$REV" > "$W/REVISION"
python3 -c "import json;t=[e for e in json.load(open('$W/tree_$REV.json')) if e['type']=='file'];print(len(t),'files',sum(e['size'] for e in t),'bytes')"   # want: 63 files 293654927078 bytes
# download in a throwaway python:3.12-slim container (pinned by digest)
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 -v "$SRC:/out" \
  python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 sh -c \
  "pip install -q --target /tmp/pkgs huggingface_hub==1.27.0 && PYTHONPATH=/tmp/pkgs python -c \
  \"from huggingface_hub import snapshot_download as s; s(repo_id='$REPO', revision='$REV', local_dir='/out', max_workers=8)\""
python3 scripts/verify_hub_v0_1_0.py --dir "$SRC" --manifests "$W"
```

- **Before the download starts, the listing check must print** `63 files 293654927078 bytes`. The listing should hold 53 safetensors shards and no `.py`, `.bin`, `.pt` or pickle files; check with `python3 -c "import json,collections;print(collections.Counter(e['path'].rsplit('.',1)[-1] for e in json.load(open('$W/tree_$REV.json')) if e['type']=='file'))"`.
- **Progress looks frozen.** `snapshot_download` writes each file only when it finishes, so watch bytes received instead (e.g. `docker stats`).
- **Pass:** `VERIFY_RESULT=PASS`. The verifier checks every file's size, the SHA-256 of the 54 large files, and the git SHA-1 of the small ones.
- **Measured:** 42 min 54 s at about 114 MB/s; verification took 73 s.

## Step 3: pad, verify and test

The padder (`scripts/pad_beam_tp3_v0_1_0.py`, SHA-256 `8a66067ee51f25b106be593245ddd1e4ce721b9e7ef54841a97dc0553c53929c`) needs only Python and NumPy.
- **Run it in the pinned vLLM image,** which supplies NumPy, with networking off and the source mounted read-only.
- **It refuses to write if it meets any tensor it does not recognise.**
- **This is the exact file that produced our checkpoint.** It is published unchanged so its hash, and the version it stamps into the output, match the checkpoint we serve. (Its header names the author's own design notes, "Beam v1.1 review (Oct 5, 2026)": a planning document written from Beam's announced architecture before the weights were released. It is not a Beam version and is not published.)
- **To re-run it, delete `$DST` first;** the padder refuses to write into an existing folder.

```bash
SRC=/srv/models/Beam-1.0-501B-A23B-NVFP4
DST=/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
VIMG=vllm/vllm-openai@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77
mkdir -p /srv/models/tp3
A="-u -I /w/pad_beam_tp3_v0_1_0.py --src $SRC --dst $DST --moe-to 2688"
docker run --rm --init --network none --user "$(id -u):$(id -g)" \
  -v "$SRC:$SRC:ro" -v /srv/models/tp3:/srv/models/tp3 -v "$PWD/scripts:/w:ro" --entrypoint sh "$VIMG" -c \
  "python3 $A --dry-run && python3 $A && python3 $A --verify && \
   python3 $A --numeric-test --numeric-layers 0,3,0 && echo ALL_GATES_PASS"
(cd "$DST" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\n' | sort | xargs -P 8 -n 1 sha256sum > SHA256SUMS)
python3 scripts/check_sums_v0_1_0.py "$DST"
```

| Phase | Pass line | Our time |
|---|---|---|
| Dry run | `PLAN_OK tensors=151262 shards=53` and `TP_GQA_MAPPING OK` | seconds |
| Write | `WRITE_DONE shards=53` | 541 s |
| Byte-level verify | `VERIFY tensors=151262 bad=0`, `VERIFY_RESULT=PASS` | 409 s |
| Numeric test | `NUMERIC_RESULT=PASS` | under a minute |
| Hash cross-check | 64 lines in `SHA256SUMS`, `SUMS_VS_MANIFEST=OK` | under a minute |

What changes, per layer (zeros appended at the end of each axis):

| Tensor | Original → padded | Zeros added to |
|---|---|---|
| `self_attn.q_proj.weight` (BF16) | [6144, 5120] → [6912, 5120] | rows (query heads 48–53) |
| `self_attn.k_proj.weight`, `v_proj.weight` (BF16) | [1024, 5120] → [1152, 5120] | rows (KV head 8) |
| `self_attn.gate_proj.weight` (BF16) | [6144, 5120] → [6912, 5120] | rows |
| `self_attn.gate_bias` (BF16) | [48] → [54] | entries |
| `self_attn.o_proj.weight` (BF16) | [5120, 6144] → [5120, 6912] | columns |
| `self_attn.q_norm`, `k_norm` | [128] | unchanged (shared across heads) |
| Dense `mlp.gate_proj`, `up_proj` (layers 0–2) | [20480, 5120] → [20736, 5120] | rows |
| Dense `mlp.down_proj` | [5120, 20480] → [5120, 20736] | columns |
| Expert `gate_proj`/`up_proj.weight` (U8, two FP4 per byte) | [2560, 2560] → [2688, 2560] | rows |
| Expert `gate_proj`/`up_proj.weight_scale` (E4M3) | [2560, 320] → [2688, 320] | rows |
| Expert `down_proj.weight` (U8) | [5120, 1280] → [5120, 1344] | byte columns |
| Expert `down_proj.weight_scale` (E4M3) | [5120, 160] → [5120, 168] | columns |
| `weight_scale_2`, `input_scale`, routers, norms, embeddings, `lm_head` | — | unchanged |

`config.json` changes in four fields only:
- `num_attention_heads` 54;
- `num_key_value_heads` 9;
- `intermediate_size` 20736;
- `moe_intermediate_size` 2688.

The padder also writes `PADDING_MANIFEST.json`, recording its version, arguments, per-shard hashes and the config changes. The proof that the padded model computes the original function is in [docs/PROOF.md](docs/PROOF.md).

## Step 4: build the serving image

The only source change is vLLM's vocabulary padding multiple, 64 → 192, so the 151,936-token vocabulary pads to 152,064 = 3 × 50,688. No fake tokens are added; vLLM masks its own padded rows.

```bash
docker build -t beam-tp3:v1 docker/
docker run --rm --network none --entrypoint python3 beam-tp3:v1 -c 'import importlib.util as u; p=u.find_spec("vllm").submodule_search_locations[0]+"/model_executor/layers/vocab_parallel_embedding.py"; print([l.strip() for l in open(p) if l.startswith("DEFAULT_VOCAB_PADDING_SIZE")])'
```

- **Pass:** prints `['DEFAULT_VOCAB_PADDING_SIZE = 192']`.
- **The build fails on purpose** if the upstream line ever changes.
- **Image IDs embed a build timestamp,** so yours will differ from ours. Compare the file content, not the ID.

## Step 5: copy to the other nodes

Both receiving nodes pull from node1 over their own ring cable at the same time: node0 over C1, node2 over C2.

```bash
NODE1=192.168.100.2        # on node0 (cable C1); on node2 use NODE1=192.168.104.1 (cable C2)
DST=/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
sudo install -d -o "$USER" -g "$USER" /srv/models/tp3
rsync -a -e "ssh -c aes128-gcm@openssh.com" <user>@$NODE1:$DST /srv/models/tp3/
ssh <user>@$NODE1 'docker save beam-tp3:v1' | docker load
(cd "$DST" && sha256sum --quiet -c SHA256SUMS && echo SUMS_MATCH)
```

- **Pass:** `SUMS_MATCH` on both receiving nodes, and the same image ID on all three (`docker image inspect -f '{{.Id}}' beam-tp3:v1`).
- **Measured on our setup, which differed:** we relayed node2's copy through node0 (only node0 held keys for both workers). The two 309 GB copies, running at once, took about 8 and 10.5 minutes. The direct copies shown above avoid sharing a cable but are untested.

## Step 6: launch

On every node:
1. Confirm no GPU processes are running (`nvidia-smi`).
2. Then run the same script with that node's rank. It refuses to start unless every check passes: image present, checkpoint present, ring ports up, no other GPU process, and at least 109 GiB free after dropping the page cache.

```bash
NODE_RANK=0 MASTER_ADDR=<node0 LAN address> bash scripts/launch_node_v0_1_4.sh   # on node0
NODE_RANK=1 MASTER_ADDR=<node0 LAN address> bash scripts/launch_node_v0_1_4.sh   # on node1
NODE_RANK=2 MASTER_ADDR=<node0 LAN address> bash scripts/launch_node_v0_1_4.sh   # on node2
```

The settings that matter:

| Setting | Why |
|---|---|
| `--nnodes 3 --node-rank R --master-addr … --distributed-executor-backend mp`, `--headless` on workers | vLLM's native multi-node mode; no Ray cluster. Rank 0 serves the API. |
| `--network host --ipc host --device /dev/infiniband --ulimit memlock=-1` | NCCL needs the host's RDMA devices and pinned memory |
| `--moe-backend b12x` | FP4 mixture-of-experts kernels for this GPU generation, shipped inside FlashInfer |
| `--kv-cache-dtype fp8` | Halves KV-cache memory; 9 KV heads cost 12.5% more than 8 would |
| `--gpu-memory-utilization 0.885` | vLLM's share of unified memory; the rest stays with the OS and host processes. 0.88 left too little KV-cache memory on two of three launches; see [Memory margin](#memory-margin) |
| `--max-model-len 131072 --max-num-seqs 4` | Our context and concurrency targets. Only 9 of 52 layers keep full-length KV; sliding-window layers keep 512 tokens. |
| `--enforce-eager` | No CUDA-graph capture: shorter start-up and less memory, slower decode. Graphs are future work. |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | Stops rank 0 giving up during the long, silent repack phase |
| `--host 127.0.0.1` | The HTTP API listens on loopback only. Put an authenticated proxy in front of it if anything else needs access. The distributed control plane (master port 29510 and the worker sockets) still listens on the LAN interface without authentication, so firewall it to the three nodes. |

**Firewall the LAN side.** An example with `ufw` (not what we ran; adapt it, and keep a console open in case you lock yourself out). On each node, allow the other two nodes' LAN addresses, the four ring interfaces and SSH from your own machine, then deny everything else:

```bash
sudo ufw allow from <other-node-A LAN address>; sudo ufw allow from <other-node-B LAN address>
for i in enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1; do sudo ufw allow in on $i; done
sudo ufw allow from <your admin machine> to any port 22 proto tcp
sudo ufw default deny incoming; sudo ufw enable
```

**Start-up time varies a lot.** With this script on Oct 21 (freshly rebooted nodes, settings as published), it took **17 minutes** from launch to `/health` returning 200, on both launches: the workers loaded their weights in about 4 minutes and node0 in about 15. Our first-ever launch (Oct 18) took an estimated 65–70 minutes, including a long phase with no log output after the weights loaded:

| First launch, Oct 18 | node0 | node1 | node2 |
|---|---|---|---|
| Weight load | 886 s | about 1,838 s | 224 s |
| One-time b12x expert repack (no log output) | about 25–30 min | about 25–30 min | about 60 min (that node was swapping) |

We have not isolated why the first launch was so much slower.

- **Do not stop the containers during a silent phase.** Allow up to about 70 minutes before suspecting a hang.
- **The script's page-cache helper:** after the weights load, it clears the Linux page cache every 10 s until vLLM has sized its KV cache, then stops. On GB10 the page cache sits in the same memory, and leftover cache from reading the weights shrinks vLLM's KV-cache estimate.
- **Ignore `NVRM` out-of-memory lines** in the kernel log during loading; they appear when `nvidia-smi` runs and do not mean loading failed. Watch `MemAvailable` falling and the container state instead.
- **Done** when `curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:8600/health` on node0 prints `200`.

## Step 7: smoke tests

On node0: `bash scripts/smoke_test_v0_1_1.sh`

| Check | Pass |
|---|---|
| Health | `200` |
| Tool call | `finish_reason=tool_calls`, one `get_weather` call with city `Paris` |
| Arithmetic canary | `391` |
| Long-context needle (synthetic service manual, 41,522 tokens; the script prints the count) | `TANGERINE-7X4Q` |

Beam reasons before it answers, and reasoning tokens count against `max_tokens`. Our first long-context probe used `max_tokens: 60` and came back empty. The script allows 600.

## Results

Measured Oct 18–21, 2026 on the hardware above.

**Pipeline time:** about 3 h 3 min from starting the download to the first token. The timed phases add up to about 2 h 20 min; the rest was the image build, checks and pauses between steps. The padding itself (write, verify and test) took about 16–17 minutes.

**Serving:**

| Probe | Result |
|---|---|
| Single-stream decode | About 18 tokens/s: 18.0–18.2 in the throughput sweep below, 18.1–18.4 in the server's own log in interactive use |
| Tool call (`smoke_test_v0_1_1.sh`, Oct 21) | `finish_reason=tool_calls`, one `get_weather` call with arguments `{"city": "Paris", "unit": "celsius"}` |
| Long-context needle (`smoke_test_v0_1_1.sh`, Oct 21) | `TANGERINE-7X4Q` returned exactly from a 41,522-token prompt |
| Arithmetic canary | `391`, 57 completion tokens (about 3.25 s in the soak's canary runs) |
| KV cache (Oct 21, published settings, two launches) | 2.44 / 1.35 / 1.82 and 2.35 / 1.36 / 1.56 GiB available on node0 / node1 / node2; the smallest sets the pool: 178,253 and 180,048 tokens, enough for 1.36–1.37 concurrent requests at the full 131,072-token context. (The first launch, at 0.88, reported 2.02 GiB per node and 266,659 tokens.) |

**Reflection's Beam Verifier (Oct 21):** `beam-verifier verify` (commit `dfa85fe771b3bcf014f5af5913f2ed363d31dfdf`), run on node0 against the loopback API. **131 of 133 checks passed** in 29 minutes, streamed and not: all response-contract (structured output) and tool-protocol checks (tool-call shape, parallel calls, every `tool_choice` mode, tool-result round trips), and every request-contract check except one. The 2 failures are the same check in both modes: a message with an invalid role (`fake_role`) should be refused with HTTP 400, but this server accepts it and starts answering (the non-streamed request then hit the verifier's time limit). vLLM's chat API accepts custom roles by design (`CustomChatCompletionMessageParam`, documented as "Enables custom roles in the Chat Completion API"), and the check runs before the model does, so we believe the padding cannot cause or fix it. We did not run the verifier against an unpadded checkpoint to confirm.
- **To run it yourself:** the verifier's documented install did not pass its options to its test runner when installed as a package, so we added one pytest setting:
  `beam-verifier verify --base-url http://127.0.0.1:8600/v1 --model-name beam -- --confcutdir="$(python -c 'import os, beam_verifier.functional_tests as f; print(os.path.dirname(f.__file__))')"` (use `python3` if `python` is not on your path)
- In that setup the verifier's summary table also shows the API suite as "SKIP — no tests ran", although the test log shows all 133 ran; the counts above are from the test log and its JSON report.

**Equivalence:** all 151,262 tensors verified byte for byte. On three sampled blocks (attention layer 0, dense layer 0, expert 0 of layer 3), numeric agreement is exact when unsharded, and within 5.3×10⁻¹⁶ relative error when split three ways ([docs/PROOF.md](docs/PROOF.md)).

**Public benchmarks (Oct 20–21, one pass each):** run with [scripts/pubbench_v1_0_5.py](scripts/pubbench_v1_0_5.py). The main runs used v1.0.0 and the re-asked questions v1.0.1 and v1.0.2 (earlier versions are not published; they hold private settings). Their question prompts and sampling settings are identical to v1.0.5's (checked by comparing the code); later versions add streaming, cap rounds, dataset sources pinned to exact commits and checked by hash, a pinned IFBench scorer, blanking of cut-off answers before scoring, and a record of the scoring containers and packages. All scores below were recomputed from the saved answers with v1.0.5; the container digests and scorer package versions it used are in each file under `results/benchmarks_20261021/`. The benchmarks ran from another machine through an authenticated, non-streaming reverse proxy, not from node0's loopback. Settings: temperature 0.7 and top_p 0.99 (Reflection's stated evaluation settings); reasoning effort not set, so Beam's chat template used its default, the maximum (Beam's chat template takes reasoning effort as a number from 8, `minimal`, to 512, `max`; higher allows longer reasoning); 4 requests at once; only the final answer scored (Beam's reasoning comes back separately and is never scored). An answer that hits its token cap is scored as wrong. The rule was fixed before scoring: a result is **consistent** if the published figure lies inside our 95% (Wilson) interval.

| Benchmark | Items | Token cap | Score | 95% interval | Published figure | Result |
|---|---|---|---|---|---|---|
| GPQA Diamond | 198 | 32,768 | **89.9%** (178) | 84.9–93.4 | 89.3% (Reflection, NVFP4) | consistent |
| GPQA Diamond, cut-off answers re-run at 65,536 | 198 | 32,768 / 65,536 | **90.9%** (180) | 86.1–94.2 | 89.3% (Reflection, NVFP4) | consistent |
| IFBench, strict | 300 | 16,384 | **75.0%** (225) | 69.8–79.6 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, lenient | 300 | 16,384 | **77.7%** (233) | 72.6–82.0 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, strict, cut-off answers re-run at 32,768 | 300 | 16,384 / 32,768 | **76.7%** (230) | 71.6–81.1 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, lenient, cut-off answers re-run at 32,768 | 300 | 16,384 / 32,768 | **79.3%** (238) | 74.4–83.5 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, strict, 3 still cut off re-run at 65,536 | 300 | 16,384 / 32,768 / 65,536 | **76.7%** (230) | 71.6–81.1 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, lenient, 3 still cut off re-run at 65,536 | 300 | 16,384 / 32,768 / 65,536 | **79.3%** (238) | 74.4–83.5 | 79.2% (Reflection, NVFP4) | consistent |
| GSM8K, first 200 | 200 | 16,384 | **97.5%** (195) | 94.3–98.9 | 97.5% (diffbot EXL3 3-bit) | consistent |
| HumanEval | 164 | 16,384 | **98.2%** (161) | 94.8–99.4 | 97.6% (diffbot EXL3 3-bit) | consistent |

- **The token cap matters.** On IFBench, 13 of 300 answers (4.3%) ran out of room at 16,384 tokens; given 32,768, 10 of those 13 finished and 3 still ran out, and 5 more prompts passed, raising both the strict and lenient scores by 1.7 points. Given 65,536, 2 of the last 3 finished but neither passed, and 1 still ran out, so the scores did not change. On GPQA, 2 of 198 ran out at 32,768; given 65,536, both finished and both were correct, raising the score by 1.0 point. In the primary rows, cut-off answers count as wrong. Reflection does not state its caps. Each re-run row keeps every other answer as it was and asks only the cut-off questions again, with a higher cap. Each re-run is a fresh sample rather than a continuation of the cut-off answer, so the difference between a benchmark's two rows is the effect of the cap plus sampling variation on those few questions. A fresh attempt can also be shorter than the one that ran out, which slightly favours the re-run row. An answer that runs out of room always scores zero, so in practice the change comes from the cut-off answers that now finish. Each further round with a higher cap re-asks only the questions still cut off, and is reported as its own row.
- **GPQA's two cut-off answers first failed** with a gateway time-out from the non-streaming proxy, because they were the two longest answers in the set. They were retried individually with the same settings and both ran to the cap. v1.0.1 streams answers, which avoids the time-out.
- Reflection does not say whether its IFBench figure is strict or lenient; we report both. "Lenient" is the scorer's `loose` mode.
- **IFBench's own guidance** is temperature 0, with the token cap chosen per model and more room for reasoning models; it sets no fixed cap. We used Reflection's stated evaluation settings instead, so at least the sampling matches theirs; Reflection does not state its caps, prompt format or reasoning effort.
- **Scorers:** IFBench with AllenAI's official code (commit `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`, pinned in v1.0.3); GPQA with OpenAI simple-evals' prompt and answer pattern, answer choices shuffled with seed 0; HumanEval by running each program against its tests in a container with no network and a read-only filesystem, 15 s limit; GSM8K by exact match on the last `Answer:` number, or on the last number in the answer if there is no `Answer:` line.
- **The GSM8K and HumanEval comparisons are rough:** they are against a different (3-bit) build whose settings are not published.
- **What these numbers do not show:** one pass each, so ordinary run-to-run variation is not measured; public benchmarks may appear in any model's training data; and matching scores are not proof of identical outputs (see [docs/PROOF.md](docs/PROOF.md) and the planned PP=3 comparison).
- **Data:** GPQA's authors ask that its questions not be posted online, so `results/benchmarks_20261021/` holds score summaries only, with no prompts or answers.

**Stability (24-hour soak, Oct 19 07:00 UTC to Oct 20 07:00 UTC):** a one-minute logger recorded `/health`, free memory, swap, NVIDIA Xid faults on every node, and server throughput, while the cluster also served other long-running requests. Result: 1,403 samples (one every 61.6 s on average) with no gap over 90 s, every sample healthy (HTTP 200), and zero Xid faults on all three nodes. A separate arithmetic canary ran every 10 minutes for 31 hours spanning the soak and passed 187 of 187, about 3.25 s each. Lowest free memory: 1 GiB, 6 GiB and 7 GiB on the three nodes; swap never exceeded 4 GiB.

**Throughput sweep (Oct 20, one run, server otherwise idle):** `vllm bench serve` from inside the serving container, random-token prompts, exactly 256 output tokens per request (`--ignore-eos`), `--seed 1`, after two unrecorded warm-up requests. 4 prompts per cell at 1 request at a time, 8 at 2, 16 at 4 (the server's `--max-num-seqs` limit), and 2 prompts of 130,000 tokens (close to the 131,072-token limit, leaving room for 256 output tokens). Median (p50) and 99th-percentile (p99) values:

| Prompt | Requests at once | Output tokens/s, all requests | Time to first token, p50 / p99 | Time per output token, p50 / p99 | Whole request, p50 |
|---|---|---|---|---|---|
| 1,024 | 1 | 17.4 | 0.72 s / 0.79 s | 54.8 ms / 54.9 ms | 14.7 s |
| 1,024 | 2 | 24.7 | 0.58 s / 1.53 s | 78.9 ms / 81.4 ms | 20.8 s |
| 1,024 | 4 | 31.9 | 0.66 s / 2.31 s | 121.5 ms / 126.1 ms | 31.8 s |
| 8,192 | 1 | 14.6 | 3.54 s / 3.66 s | 55.0 ms / 55.2 ms | 17.6 s |
| 8,192 | 2 | 20.5 | 3.66 s / 4.86 s | 90.0 ms / 93.6 ms | 25.8 s |
| 8,192 | 4 | 26.5 | 3.81 s / 10.86 s | 132.1 ms / 156.0 ms | 37.4 s |
| 32,768 | 1 | 9.6 | 12.51 s / 13.26 s | 55.1 ms / 55.3 ms | 26.6 s |
| 32,768 | 2 | 14.0 | 8.64 s / 31.72 s | 81.8 ms / 137.5 ms | 30.4 s |
| 32,768 | 4 | 16.4 | 8.96 s / 61.32 s | 125.6 ms / 288.0 ms | 40.6 s |
| 130,000 | 1 | 3.0 | 71.69 s / 71.76 s | 55.7 ms / 55.9 ms | 85.9 s |

What this means (derived from the table, not measured separately):

- **One request at a time decodes at about 18 tokens/s** (1000 ÷ 54.8 ms), and that barely changes with context: 18.0 tokens/s at 130,000 tokens.
- **Reading the prompt (prefill) runs at about 2,300 tokens/s at 8K, 2,600 at 32K and 1,800 at 130K** (prompt length ÷ time to first token at 1 request). A full-length prompt takes a little over a minute before the first token.
- **Four requests at once give about 1.8× the total output of one** (31.9 vs 17.4 tokens/s at 1K), but each request slows to about 8 tokens/s.
- **A long prompt arriving stalls the others.** With long prompts and more than one request, the slowest 1% of gaps between tokens reached about 1 second (raw `ITL` p99 in the result files: 0.93–1.11 s), because the new prompt is read while the others wait.
- The output column counts the whole run time, including prefill, so it falls as prompts get longer even though decoding speed does not.

Limits of these numbers: small samples (2–16 requests per cell, so p99 is essentially the slowest request), random tokens rather than real text, fixed 256-token answers, and a single run. Raw per-cell JSON is in `results/sweep_20261020/`.

**Cost of padding** (from tensor shapes, not measured):

| Component | Original | Padded | Growth |
|---|---|---|---|
| Attention weights, BF16, 52 layers | 10.90 GB | 12.27 GB | +12.5% |
| Expert weights incl. block scales | 277.45 GB | 291.33 GB | +5.0% |
| Dense feed-forward, 3 layers | 1.887 GB | 1.911 GB | +1.25% |
| Checkpoint total | 293.65 GB | ≈309 GB | +5.2% |
| Weight bytes read per decoded token | ≈23.0 GB | ≈24.8 GB | ≈+8% |
| KV cache per token | 8 heads | 9 heads | +12.5% |

Each node holds about 103 GB of weights.

### Memory margin

vLLM gives itself `GPU_UTIL` × 121.69 GiB on each node (107.7 GiB at 0.885). Weights, start-up working memory and other non-KV memory took 105.26–105.83 GiB per node on Oct 21; the KV cache gets the rest, and the smallest node's figure decides. One 131,072-token request needs 0.99 GiB.

| Launch | `GPU_UTIL` | KV-cache memory, node0 / node1 / node2 | Result |
|---|---|---|---|
| Oct 18 (first) | 0.88 | 2.02 GiB per node | started |
| Oct 21, 17:16 | 0.88 | 0.8 GiB on node0 (others not recorded) | failed: too little KV memory |
| Oct 21, 20:46 (page-cache helper added) | 0.88 | 1.42 / 0.77 / 0.91 GiB | failed: too little KV memory on node1 and node2 |
| Oct 21, 21:15 (published settings) | **0.885** | **2.44 / 1.35 / 1.82 GiB** | **started**; smallest node 36% above what 131K needs |
| Oct 21, 21:47 (published settings, repeat) | **0.885** | **2.35 / 1.36 / 1.56 GiB** | **started**; 180,048-token pool |

- The figure swings by up to about 1.2 GiB between launches of identical settings. Leftover page cache is part of it (the helper raised node0 from 0.8 to 1.42 GiB at the same setting); the rest we have not pinned down. Node1 also runs unrelated services in our setup.
- **If you hit the error:** raise `GPU_UTIL` to 0.89 (+1.2 GiB per node, taken from the OS; vLLM refuses to start if free memory at launch is below `GPU_UTIL` × 121.69 GiB, about 108.3 GiB at 0.89), or lower `MAX_MODEL_LEN` to the estimate vLLM prints. Freeing memory on the nodes (for example, running without a desktop session, about 2–3 GB) also helps.
- **Stopping does not give the memory back.** After `docker rm -f beam-tp3`, whether the server had crashed or was running normally, about 105 GiB per node stayed in use with no process holding it (only the NVIDIA persistence daemon and the desktop had the GPU open), and restarting `nvidia-persistenced` did not release it. **Reboot every node before relaunching.** Whether a graceful `docker stop -t 120` avoids this is untested.

## Why it is built this way

| Choice | Problem it solves | Cost |
|---|---|---|
| **Tensor parallelism (TP=3), not pipeline parallelism (PP=3)** | Single-stream decode is limited by memory bandwidth (about 273 GB/s per GB10). With PP, the three stages run one after another for every token, so one stream only ever uses one node's bandwidth. With TP, all three read their third of each layer at once. | At least two all-reduces per layer (104 per token) over the ring. We have not yet measured PP=3 on this model. |
| **Pad the checkpoint offline, not at load time** | Stock vLLM asserts that heads and widths divide by the TP degree. Padding the files means a stock image works, and the result can be hashed, verified byte for byte and inspected. | 5.2% more disk and memory, about 8% more bytes read per token |
| **54 query heads, although 48 divides by 3** | KV heads must become 9. Each query head reads KV head ⌊h/6⌋, and keeping the 6:1 ratio is what lets every rank hold whole query-KV groups. vLLM's alternative, duplicating KV heads, only applies when TP is a multiple of the KV count. | 12.5% more attention weights and KV cache |
| **Expert width 2,688, not the smallest valid 2,592** | 2,688 gives 896 per rank, a multiple of 64. vLLM's FlashInfer NVFP4 MoE path pads per-rank width to 64 at load time anyway. We did not confirm b12x's own rule, so this is a precaution, not a proven requirement. | 5.0% extra expert storage instead of 1.25%; extra compute only if the b12x path does not pad per-rank widths itself; 2,592 is untested |
| **Dense width 20,736, not the smallest valid 20,496** | 6,912 per rank is a multiple of 128, which common matrix-multiply tiles divide evenly. We did not measure whether that matters. | 1.25% on three layers |
| **Change vLLM's vocabulary padding instead of the vocabulary** | The tokenizer and embedding tensors stay untouched, and no fake tokens exist | A one-line image patch |
| **FP8 KV cache** | Halves KV memory on nodes with little room left after 103 GB of weights | Small numeric change in attention |
| **`--enforce-eager`** | Faster start-up and less memory | Slower decode |
| **Native multi-node vLLM, no Ray** | Fewer moving parts; Ray's memory reservations matter on unified memory | — |
| **Pinned digests, throwaway download container, read-only source, no network for the padder** | The model, base image, download environment, host test tools, nccl-tests and benchmark datasets are pinned, and nothing in the build can change the source or reach the internet. The benchmark containers are recorded by digest in the results rather than fixed in the script. | Slightly longer commands |
| **HTTP API on loopback only** | The model API is not exposed by default | Needs a proxy for remote access. The distributed control plane on the LAN is unauthenticated; firewall it to the three nodes. |
| **Reflection's own NVFP4 checkpoint, not a community re-quantization** | Vendor-calibrated weights with published benchmarks to compare against | Larger than 3-bit alternatives (see [Related work](#related-work-and-other-builds)) |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Padder stops with a "file exists" error | The output folder is left over from an earlier run | Delete `$DST` and run Step 3 again |
| Out of disk during download or padding | Original plus padded copy plus images need about 650 GB | Build on the node with the most free disk |
| Download runs at about 12 MB/s | That node's management link negotiated 100 Mb/s | Download on a node with a fast uplink, then copy over the ring |
| Download progress looks frozen | `snapshot_download` writes each file only when it finishes | Watch bytes received, not files on disk |
| `ModuleNotFoundError: numpy` on the host | DGX OS ships without NumPy | Run the padder inside the vLLM image, as shown |
| Pair tests fail over one cable | The cable joins two same-numbered ports | Re-cable port 0 to port 1 (NVIDIA's wiring) |
| `STOP: image beam-tp3:v1 not found` | The launch script expects the image under that name | Build it with Step 4, or tag your existing image: `docker tag <image id> beam-tp3:v1` |
| `ValueError: To serve at least one request with the model's max seq len (131072), (0.99 GiB KV cache is needed, which is larger than the available KV cache memory …` | vLLM's KV-cache estimate came out too small on at least one node | See [Memory margin](#memory-margin): raise `GPU_UTIL` by 0.005, or set `MAX_MODEL_LEN` to the "estimated maximum model length" the error prints |
| After stopping or a failed start, `free -g` shows about 110 GiB used with no container and no GPU process | The driver does not return the memory of killed processes on our nodes | Reboot every node before relaunching |
| Long silence after the weights load | One-time in-process b12x expert repack | Wait. Keep `VLLM_ENGINE_READY_TIMEOUT_S=3600`. Don't lower memory or context limits in response. |
| `NVRM` out-of-memory lines during start-up | Printed when `nvidia-smi` runs during loading | Watch `MemAvailable` and container state instead |
| Empty answer to a long prompt | Reasoning used the whole `max_tokens` budget | Allow at least 600 tokens |
| A bind-mounted file appears as a root-owned folder | Docker creates missing mount sources as folders | Create mounted files and folders before `docker run` |

## Limitations

1. **No full-model equivalence measurement yet.** The proof is analytic and the numeric test samples three blocks: one attention layer (layer 0), one dense layer and one expert; other attention-layer types were not sampled separately. Planned: serve the unpadded checkpoint at PP=3 (pipeline stages split by layer, so no divisibility problem) and compare per-token probabilities with this build at temperature 0. The same run gives a measured TP-versus-PP speed comparison.
2. **Quality is measured by four public benchmarks, one pass each.** GPQA Diamond and IFBench are consistent with Reflection's published NVFP4 figures; GSM8K and HumanEval match a community 3-bit build. They do not measure quality on any particular real-world task, and run-to-run variation is not measured.
3. **Eager mode only, one sweep.** All speed figures are with `--enforce-eager` (CUDA graphs off) from a single run with small samples. With several requests and long prompts, a newly arriving prompt pauses the others for up to about a second. Not tried: CUDA graphs, chunked-prefill tuning, speculative decoding.
4. **Tight memory.** See [Memory margin](#memory-margin). After loading, one node had 2 GiB free and every node carried 3–4 GiB of swap (24-hour soak, at 0.88). Concurrency × context is bounded by 178,253 KV tokens. Long-run stability at 0.885 has not yet been soak-tested.
5. **Kernel behaviour on all-zero blocks is assumed, not proven.** The proof relies on each kernel mapping an all-zero FP4 block to zero. Outputs were finite and correct, but we did not audit every kernel.
6. **Unexplained start-up variance.** One node loaded identical files about 8× slower than another (1,838 s against 224 s).
7. **A slow management link.** One worker's LAN link negotiated 100 Mb/s. vLLM sends rendezvous and per-step scheduler messages to each worker over that link; its effect on decode latency is unmeasured.
8. **Invalid message roles are accepted.** Reflection's verifier expects HTTP 400 for a message with an unknown role; vLLM's chat API accepts custom roles by design, so this build does too.
9. **One repetition loop** appeared once in a chat session using tools and has not been root-caused. A candidate fix is the server variable `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT=0`.
10. **n = 1:** one cluster, one launch.

## Related work and other builds

**TP=3 on DGX Spark, other models.** All of these patch or fork vLLM, or pad at load time; this build pads the files and serves on the pinned upstream image plus a one-constant patch.
- [MiniMax M3 on 3 Sparks TP=3](https://forums.developer.nvidia.com/t/minimax-m3-on-3-sparks-tp-3-is-now-working/373388) (Jun 15, 2026): "virtual sharding" in Luke Alonso's vLLM fork, plus b12x.
- [GLM-5.3-Flash NVFP4 on 3× DGX Spark TP=3](https://forums.developer.nvidia.com/t/glm-5-3-flash-nvfp4-on-3x-dgx-spark-tp-3-512k-context-35-tok-s/381534) (Aug 27, 2026): heads and experts padded via vLLM overlay files.
- [DeepSeek-V4-Flash TP3 on 3× DGX Spark](https://huggingface.co/PrblyElect/DeepSeek-V4-Flash-TP3-3x-DGX-Spark) (Oct 2026): padding in a patched or forked vLLM.

**The padding technique.** [marianbusoi/Hy3-NVFP4-FP8-TP3](https://huggingface.co/marianbusoi/Hy3-NVFP4-FP8-TP3) zero-padded attention heads and the dense layer of another large MoE for TP=3, and patched the vocabulary multiple the same way. Beam additionally needs its gated attention and its NVFP4 experts (with their FP8 block scales) padded.

**Other Beam builds** (as of Oct 21, 2026):
- **[Baekpica/Beam-1.0-501B-A23B-VQ-GGUF](https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF)** (Oct 20, 2026): a Beam build for a **single** DGX Spark. Routed experts are vector-quantized to about 1.5 bits per weight (about 100 GB in total), served by its own engine, [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs). It reports about 25 tokens/s decode and labels itself experimental, with reasoning loops and broken output possible. It trades fidelity for fitting on one node; this repo keeps Reflection's NVFP4 weights unchanged apart from zero padding, and spreads them over three.
- **[diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000)**
  - 3-bit experts and 6-bit attention in EXL3, 194 GB, TP=2 on two RTX PRO 6000 cards, served by a patched vLLM with ExLlamaV3 kernels.
  - It reports divergence from the full-precision model (KL divergence on its test set) of 0.0132 for itself and 0.0116 for NVFP4 as served on Blackwell. That is a useful outside reference for how close a served NVFP4 Beam is to the original.
  - Its card states that its vllm-exl3 patch is AGPL-3.0-only.
- **GGUF quantizations** for llama.cpp: [unsloth](https://huggingface.co/unsloth/Beam-1.0-501B-A23B-GGUF) and [AtomicChat](https://huggingface.co/AtomicChat/Beam-1.0-501B-A23B-GGUF), which publishes KL-divergence figures for each size.
- The official [vLLM recipe](https://recipes.vllm.ai/reflection/Beam-1.0-501B-A23B) covers B200/B300 and GB200/GB300 at TP4, and H100/H200 at TP8; the [day-0 blog post](https://vllm.ai/blog/2026-10-16-reflection-beam) covers GB200/GB300 at TP4/TP8. Neither covers DGX Spark or TP=3. Reflection's own [deployment guides](https://developers.reflection.ai/deployment-overview) and [cookbook](https://github.com/reflection-oss/cookbook) cover data-centre serving and fine-tuning, not DGX Spark.
- A [tuned configuration for 4× RTX PRO 6000](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4/discussions/5) on a single PCIe workstation is posted in the NVFP4 model's discussions.

## Credits, citation and licence

**Built on:**
- [Reflection](https://reflection.ai/blog/introducing-beam): the model.
- [vLLM](https://github.com/vllm-project/vllm): serving.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and **Luke Alonso's [b12x](https://pypi.org/project/b12x/)** kernels, which ship inside FlashInfer: the FP4 MoE path.
- NVIDIA's [three-Spark ring playbook](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring): the wiring.
- **eugr and dbsci** on the NVIDIA developer forum: the switchless-ring NCCL settings this build uses ([Mar 31](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160) and [Apr 1, 2026](https://forums.developer.nvidia.com/t/three-node-spark-clusters-without-a-switch-are-now-supported-in-spark-vllm-docker-and-sparkrun/365296)).
- **marianbusoi:** the Hy3 TP=3 padding approach.
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) and [perftest](https://github.com/linux-rdma/perftest).
- Benchmarks: [GPQA](https://arxiv.org/abs/2311.12022) (Rein et al.), [IFBench](https://github.com/allenai/IFBench) (Pyatkin et al., AllenAI), [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al.), [HumanEval](https://arxiv.org/abs/2107.03374) (Chen et al.), and OpenAI's [simple-evals](https://github.com/openai/simple-evals) for the GPQA prompt and answer format.

**Citing Beam** (Reflection's requested citation):

```bibtex
@misc{reflection2026introducingbeam,
  author = {{Reflection}},
  title  = {Introducing {Beam}: {Reflection}'s 501B open-weight model},
  year   = {2026},
  month  = oct,
  url    = {https://reflection.ai/blog/introducing-beam}
}
```

To cite this repo, use GitHub's "Cite this repository" button, which reads [CITATION.cff](CITATION.cff).

**How this was made:** architected, orchestrated, reviewed, approved, tested and validated by Josh Pearlman on his own three-node DGX Spark cluster. Code and documentation drafted with Claude (Anthropic).

**Licence:**
- Code (`scripts/`, `docker/`, `CITATION.cff`, `REUSE.toml`): [Apache-2.0](LICENSES/Apache-2.0.txt).
- Documentation and results (`README.md`, `docs/`, `results/`): [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Command examples in the documentation may also be used under Apache-2.0.
- The Hugging Face model card and notice (`hf/`): Apache-2.0, matching the model repository they ship in.
- See [REUSE.toml](REUSE.toml).
- Beam's weights are Apache-2.0 from Reflection. This repo distributes no weights; the padded checkpoint is on Hugging Face (see Step 2).

Provided as is, with no warranty. This setup runs each node close to its memory limit.

**Support:** issues are open; help is best effort, with no support promised.
