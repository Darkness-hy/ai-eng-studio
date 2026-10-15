from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .code_rag_benchmark import CHUNKS, QUERIES
from .portfolio_benchmarks import RawCaseResult

VARIANTS = ("bm25_body", "bm25_ast", "hybrid", "hybrid_rerank")
CHUNK_INDEX_BY_ID = {str(chunk["id"]): index for index, chunk in enumerate(CHUNKS)}


def _identifier_tokens(text: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", "-")
    return re.findall(r"[a-z0-9]+", expanded.lower())


def _field_tokens(chunk: dict[str, Any], field: str) -> list[str]:
    values = chunk["ast"].get(field, [])
    return _identifier_tokens(" ".join(str(item) for item in values))


def _ast_text(chunk: dict[str, Any]) -> str:
    symbols = _field_tokens(chunk, "symbols")
    calls = _field_tokens(chunk, "calls")
    arguments = _field_tokens(chunk, "arguments")
    imports = _field_tokens(chunk, "imports")
    path = _identifier_tokens(str(chunk["path"]))
    structural = symbols * 4 + calls * 2 + arguments + imports + path * 2
    return f"{' '.join(structural)} {chunk['code']}"


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokens = [_identifier_tokens(document) for document in documents]
        self.lengths = [len(item) for item in self.tokens]
        self.avg_length = statistics.fmean(self.lengths)
        self.term_frequencies = [Counter(item) for item in self.tokens]
        self.document_frequencies: Counter[str] = Counter()
        for counts in self.term_frequencies:
            self.document_frequencies.update(counts.keys())

    def search(self, query: str) -> list[tuple[int, float]]:
        scores = [0.0] * len(self.tokens)
        document_count = len(self.tokens)
        for term in _identifier_tokens(query):
            frequency = self.document_frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                (document_count - frequency + 0.5) / (frequency + 0.5) + 1
            )
            for index, counts in enumerate(self.term_frequencies):
                term_count = counts.get(term, 0)
                if not term_count:
                    continue
                denominator = term_count + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.avg_length
                )
                scores[index] += (
                    inverse_document_frequency * term_count * (self.k1 + 1) / denominator
                )
        return sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))


def _dense_rankings(documents: list[str], queries: list[str]) -> list[list[tuple[int, float]]]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_features=30_000,
        dtype=np.float32,
    )
    document_matrix = vectorizer.fit_transform(documents)
    components = min(64, document_matrix.shape[0] - 1, document_matrix.shape[1] - 1)
    reducer = TruncatedSVD(n_components=components, random_state=0)
    normalizer = Normalizer(copy=False)
    dense_documents = normalizer.fit_transform(reducer.fit_transform(document_matrix))
    dense_queries = normalizer.transform(reducer.transform(vectorizer.transform(queries)))
    rankings: list[list[tuple[int, float]]] = []
    for query_vector in dense_queries:
        scores = dense_documents @ query_vector
        rankings.append(sorted(enumerate(scores.tolist()), key=lambda item: (-item[1], item[0])))
    return rankings


def _rrf(
    sparse: list[tuple[int, float]], dense: list[tuple[int, float]], constant: int = 60
) -> list[tuple[int, float]]:
    scores: defaultdict[int, float] = defaultdict(float)
    for rank, (index, _score) in enumerate(sparse):
        scores[index] += 1 / (constant + rank + 1)
    for rank, (index, _score) in enumerate(dense):
        scores[index] += 1 / (constant + rank + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _rerank(query: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
    query_tokens = set(_identifier_tokens(query))
    reranked: list[tuple[int, float]] = []
    for prior_rank, (index, _score) in enumerate(candidates):
        chunk = CHUNKS[index]
        symbols = set(_field_tokens(chunk, "symbols"))
        calls = set(_field_tokens(chunk, "calls"))
        arguments = set(_field_tokens(chunk, "arguments"))
        imports = set(_field_tokens(chunk, "imports"))
        path = set(_identifier_tokens(str(chunk["path"])))
        score = (
            1 / (prior_rank + 1)
            + 0.55 * len(query_tokens & symbols)
            + 0.25 * len(query_tokens & calls)
            + 0.10 * len(query_tokens & arguments)
            + 0.07 * len(query_tokens & imports)
            + 0.03 * len(query_tokens & path)
        )
        reranked.append((index, score))
    return sorted(reranked, key=lambda item: (-item[1], item[0]))


def _dcg(relevances: list[float]) -> float:
    return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(relevances))


def _score_query(
    query: dict[str, Any], ranked_indices: list[int], latency_ms: float
) -> RawCaseResult:
    judgments = {str(item["chunkId"]): float(item["relevance"]) for item in query["judgments"]}
    raw_top_indices = ranked_indices[:10]
    raw_top_ids = [str(CHUNKS[index]["id"]) for index in raw_top_indices]
    judged_indices = [
        index for index in ranked_indices if str(CHUNKS[index]["id"]) in judgments
    ][:10]
    ranked_ids = [str(CHUNKS[index]["id"]) for index in judged_indices]
    grades = [judgments[chunk_id] for chunk_id in ranked_ids]
    ideal_grades = sorted(judgments.values(), reverse=True)[:10]
    ideal_dcg = _dcg(ideal_grades)
    positive_ids = {chunk_id for chunk_id, relevance in judgments.items() if relevance > 0}
    reciprocal_rank = next(
        (1 / rank for rank, relevance in enumerate(grades, start=1) if relevance > 0),
        0.0,
    )
    anchors = tuple(
        f"{CHUNKS[index]['repo']}/{CHUNKS[index]['path']}:"
        f"{CHUNKS[index]['startLine']}-{CHUNKS[index]['endLine']}"
        for index in raw_top_indices[:5]
    )
    return RawCaseResult(
        case_id=str(query["id"]),
        primary_score=_dcg(grades) / ideal_dcg if ideal_dcg else 0,
        secondary={
            "mrr_at_10": reciprocal_rank,
            "recall_at_10": len(positive_ids & set(raw_top_ids)) / len(positive_ids),
            "citation_precision": sum(
                judgments.get(chunk_id, 0) > 0 for chunk_id in raw_top_ids[:3]
            )
            / 3,
        },
        evidence=anchors,
        latency_ms=latency_ms,
    )


def run_code_rag_variant(variant: str) -> list[RawCaseResult]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported CodeSearchNet variant: {variant}")
    ast_documents = [_ast_text(chunk) for chunk in CHUNKS]
    documents = [str(chunk["code"]) for chunk in CHUNKS] if variant == "bm25_body" else ast_documents
    bm25 = BM25Index(documents)
    query_texts = [str(query["text"]) for query in QUERIES]
    dense = _dense_rankings(ast_documents, query_texts) if variant in {"hybrid", "hybrid_rerank"} else None
    results: list[RawCaseResult] = []
    for query_index, query in enumerate(QUERIES):
        started = time.perf_counter()
        sparse_ranking = bm25.search(str(query["text"]))
        ranking = _rrf(sparse_ranking, dense[query_index]) if dense is not None else sparse_ranking
        if variant == "hybrid_rerank":
            ranking = _rerank(str(query["text"]), ranking[:100])
        ranked_indices = [index for index, _score in ranking]
        results.append(_score_query(query, ranked_indices, (time.perf_counter() - started) * 1000))
    return results
