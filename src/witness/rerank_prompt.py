"""Qwen3-Reranker-0.6B yes/no scoring prompt (official recipe), identical to the DuRetrieval runs.

Score s = softmax([logit_no, logit_yes])[1] at the last position.
"""
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
          "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
DOC_CHARS = 600
MAX_LEN = 512


def build(query, doc):
    return f"{PREFIX}<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {doc[:DOC_CHARS]}{SUFFIX}"
