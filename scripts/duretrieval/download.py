"""Download the DuRetrieval dataset repos into {duretrieval_dir}/hf and the model repos of the D3 systems into
{models_dir}, with huggingface_hub.snapshot_download.

The Hub endpoint is huggingface_hub's own: https://huggingface.co unless the user exports HF_ENDPOINT (e.g. a mirror).
Model downloads skip the ONNX / TF / Flax / OpenVINO / CoreML exports, which no script here loads.
usage: HF_HUB_DISABLE_XET=1 PYTHONPATH=src python scripts/duretrieval/download.py configs/d3_duretrieval.yaml {datasets|models}
"""
import sys

from huggingface_hub import snapshot_download
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATASETS = ["C-MTEB/DuRetrieval", "C-MTEB/DuRetrieval-qrels"]
# the 10 dense encoders of scripts/duretrieval/encode.py and the reranker of scripts/duretrieval/rerank.py
MODELS = ["BAAI/bge-m3", "intfloat/multilingual-e5-small", "Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-small-zh-v1.5",
          "BAAI/bge-base-zh-v1.5", "thenlper/gte-small-zh", "thenlper/gte-base-zh", "moka-ai/m3e-base",
          "intfloat/multilingual-e5-base", "shibing624/text2vec-base-chinese", "Qwen/Qwen3-Reranker-0.6B"]
IGNORE = ["*.onnx", "onnx/*", "*.msgpack", "*.h5", "flax_model*", "tf_model*", "openvino/*", "coreml/*", "*.ot", "imgs/*"]

cfg = load_config(sys.argv[1])
what = sys.argv[2]
assert what in ("datasets", "models"), what
if what == "datasets":
    for repo in DATASETS:
        path = snapshot_download(repo, repo_type="dataset", local_dir=f"{cfg['duretrieval_dir']}/hf/{repo.split('/')[1]}")
        print("DATA", repo, path)
else:
    for repo in MODELS:
        path = snapshot_download(repo, local_dir=f"{cfg['models_dir']}/{repo.split('/')[1]}", ignore_patterns=IGNORE)
        print("MODEL", repo, path)
print("DONE")
