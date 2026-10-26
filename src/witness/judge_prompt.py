"""Relevance-judging prompt shared by every LLM judge (verbatim copy of the DuRetrieval runs' strict prompt,
so that cache keys sha1(model|system|prompt) and labels stay comparable across datasets)."""

SYSTEM = "你是严格的检索相关性标注员，只依据给定段落本身作答，不使用外部知识。"
TEMPLATE = (
    "查询：{q}\n段落：{p}\n\n"
    "请判断该段落是否包含能直接回答该查询信息需求的内容。只输出一行 JSON：{{\"label\": 0或1或2, \"reason\": \"不超过30字\"}}。\n"
    "0=不相关，或只是主题相关但不含答案；1=部分回答（含相关信息但不完整、不够具体或需要推断）；2=直接且完整地回答。"
)
PASSAGE_CHARS = 800

# English version for D2 (TREC DL 2019): a direct translation of the Chinese prompt with the same 0/1/2 scale.
# The passage cap is larger because MS MARCO passages are English and 800 characters would truncate some of them.
SYSTEM_EN = ("You are a strict retrieval relevance annotator. Answer based only on the given passage itself; "
             "do not use external knowledge.")
TEMPLATE_EN = (
    "Query: {q}\nPassage: {p}\n\n"
    "Determine whether the passage contains content that directly answers the information need of the query. "
    "Output exactly one line of JSON: {{\"label\": 0 or 1 or 2, \"reason\": \"at most 20 words\"}}.\n"
    "0 = not relevant, or only topically related without containing the answer; "
    "1 = partial answer (contains relevant information but is incomplete, not specific enough, or requires inference); "
    "2 = directly and completely answers the query."
)
PASSAGE_CHARS_EN = 2000
