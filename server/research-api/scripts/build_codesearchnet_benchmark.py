from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import re
import textwrap
import time
import tokenize
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/github/CodeSearchNet"
SOURCE_COMMIT = "106e827405c968597da938f6b373d30183918869"
ANNOTATION_PATH = "resources/annotationStore.csv"
ANNOTATION_URL = f"{SOURCE_REPOSITORY}/raw/{SOURCE_COMMIT}/{ANNOTATION_PATH}"
ANNOTATION_SHA256 = "0340af32b551ceadb74fec147f97642b7fedf3ff039e38fb86baff49ee899846"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research_api"
    / "data"
    / "codesearchnet_python_human_v1.json.gz"
)
URL_PATTERN = re.compile(
    r"https://github\.com/([^/]+/[^/]+)/blob/([0-9a-f]{40})/(.+)#L(\d+)-L(\d+)$"
)


class DocstringStripper(ast.NodeTransformer):
    @staticmethod
    def _strip(node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._strip(node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._strip(node)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._strip(node)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self._strip(node)
        return self.generic_visit(node)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def download(url: str, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    request = urllib.request.Request(url, headers={"User-Agent": "ai-eng-studio-benchmark-builder/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_error = exc
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(0.4 * (attempt + 1))
    assert last_error is not None
    raise last_error


def load_annotations(path: Path | None) -> bytes:
    payload = path.read_bytes() if path else download(ANNOTATION_URL)
    digest = sha256_bytes(payload)
    if digest != ANNOTATION_SHA256:
        raise RuntimeError(f"annotation SHA-256 mismatch: expected {ANNOTATION_SHA256}, got {digest}")
    return payload


def parse_url(url: str) -> dict[str, Any]:
    match = URL_PATTERN.fullmatch(url)
    if not match:
        raise ValueError(f"unsupported CodeSearchNet URL: {url}")
    repository, commit, path, start_line, end_line = match.groups()
    return {
        "repository": repository,
        "commit": commit,
        "path": path,
        "startLine": int(start_line),
        "endLine": int(end_line),
    }


def raw_url(source: dict[str, Any]) -> str:
    path = urllib.parse.quote(str(source["path"]), safe="/")
    return (
        f"https://raw.githubusercontent.com/{source['repository']}/"
        f"{source['commit']}/{path}"
    )


def strip_comments_fallback(source: str) -> str:
    output: list[tokenize.TokenInfo] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                continue
            output.append(token)
        return tokenize.untokenize(output).strip()
    except (IndentationError, tokenize.TokenError):
        return source.strip()


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def parse_python(source: str) -> tuple[str, dict[str, Any]]:
    dedented = textwrap.dedent(source).strip()
    tree = ast.parse(dedented)
    stripped_tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(stripped_tree)
    code = ast.unparse(stripped_tree).strip()

    symbols: set[str] = set()
    arguments: set[str] = set()
    calls: set[str] = set()
    imports: set[str] = set()
    node_types: set[str] = set()
    for node in ast.walk(stripped_tree):
        node_types.add(type(node).__name__)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            symbols.add(node.name)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            arguments.update(argument.arg for argument in node.args.args)
            arguments.update(argument.arg for argument in node.args.kwonlyargs)
            if node.args.vararg:
                arguments.add(node.args.vararg.arg)
            if node.args.kwarg:
                arguments.add(node.args.kwarg.arg)
        if isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name:
                calls.add(name)
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            imports.update(f"{prefix}.{alias.name}".strip(".") for alias in node.names)
    return code, {
        "symbols": sorted(symbols),
        "arguments": sorted(arguments),
        "calls": sorted(calls),
        "imports": sorted(imports),
        "nodeTypes": sorted(node_types),
    }


def fetch_source(item: tuple[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    key, source = item
    repository, commit, path = key
    try:
        payload = download(raw_url(source))
        return {
            "ok": True,
            "key": key,
            "fileSha256": sha256_bytes(payload),
            "fileText": payload.decode("utf-8", errors="replace"),
            "repository": repository,
            "commit": commit,
            "path": path,
        }
    except Exception as exc:
        return {
            "ok": False,
            "key": key,
            "errorType": type(exc).__name__,
            "error": str(exc)[:300],
        }


def split_for_query(query: str) -> str:
    bucket = int(hashlib.sha256(query.encode()).hexdigest()[:8], 16) % 5
    return "dev" if bucket == 0 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen CodeSearchNet Python benchmark")
    parser.add_argument("--annotation-file", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    annotation_payload = load_annotations(args.annotation_file)
    rows = list(csv.DictReader(io.StringIO(annotation_payload.decode("utf-8-sig"))))
    python_rows = [row for row in rows if row["Language"].strip().lower() == "python"]

    pair_scores: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    source_by_url: dict[str, dict[str, Any]] = {}
    file_sources: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in python_rows:
        query = row["Query"].strip().lower()
        url = row["GitHubUrl"].strip()
        pair_scores[(query, url)].append(float(row["Relevance"]))
        source = parse_url(url)
        source_by_url[url] = source
        file_sources[(source["repository"], source["commit"], source["path"])] = source

    fetched: dict[tuple[str, str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    items = sorted(file_sources.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(fetch_source, items):
            if result["ok"]:
                fetched[tuple(result["key"])] = result
            else:
                failures.append(result)

    chunks: list[dict[str, Any]] = []
    chunk_id_by_url: dict[str, str] = {}
    snippet_failures: list[dict[str, Any]] = []
    for url, source in sorted(source_by_url.items()):
        file_result = fetched.get((source["repository"], source["commit"], source["path"]))
        if not file_result:
            continue
        lines = file_result["fileText"].splitlines()
        start = int(source["startLine"])
        end = int(source["endLine"])
        if start < 1 or end < start or end > len(lines):
            snippet_failures.append(
                {
                    "url": url,
                    "errorType": "InvalidLineRange",
                    "lineCount": len(lines),
                }
            )
            continue
        raw_code = "\n".join(lines[start - 1 : end]).strip()
        try:
            code, ast_fields = parse_python(raw_code)
            if not code:
                raise ValueError("empty code after docstring removal")
        except Exception as exc:
            snippet_failures.append(
                {
                    "url": url,
                    "errorType": type(exc).__name__,
                }
            )
            continue
        chunk_id = f"csn.{hashlib.sha256(url.encode()).hexdigest()[:20]}"
        chunk_id_by_url[url] = chunk_id
        chunks.append(
            {
                "id": chunk_id,
                "repo": source["repository"],
                "commit": source["commit"],
                "path": source["path"],
                "startLine": source["startLine"],
                "endLine": source["endLine"],
                "url": url,
                "fileSha256": file_result["fileSha256"],
                "rawCode": raw_code,
                "rawCodeSha256": sha256_bytes(raw_code.encode()),
                "code": code,
                "codeSha256": sha256_bytes(code.encode()),
                "ast": ast_fields,
            }
        )

    judgments_by_query: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    unavailable_pairs = 0
    for (query, url), scores in sorted(pair_scores.items()):
        chunk_id = chunk_id_by_url.get(url)
        if not chunk_id:
            unavailable_pairs += 1
            continue
        judgments_by_query[query].append(
            {
                "chunkId": chunk_id,
                "relevance": sum(scores) / len(scores),
                "annotationCount": len(scores),
            }
        )

    queries: list[dict[str, Any]] = []
    dropped_queries: list[dict[str, Any]] = []
    for query, judgments in sorted(judgments_by_query.items()):
        positive = [item for item in judgments if item["relevance"] > 0]
        strongly_relevant = [item for item in judgments if item["relevance"] >= 2]
        if len(judgments) < 7 or not positive:
            dropped_queries.append(
                {
                    "query": query,
                    "judgmentCount": len(judgments),
                    "positiveCount": len(positive),
                    "stronglyRelevantCount": len(strongly_relevant),
                }
            )
            continue
        queries.append(
            {
                "id": f"q{len(queries) + 1:03d}",
                "text": query,
                "split": split_for_query(query),
                "judgments": sorted(judgments, key=lambda item: item["chunkId"]),
            }
        )

    payload = {
        "schemaVersion": "2.0.0",
        "source": {
            "name": "CodeSearchNet Challenge human relevance judgments",
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "annotationPath": ANNOTATION_PATH,
            "annotationSha256": ANNOTATION_SHA256,
            "language": "Python",
            "relevanceAggregation": "arithmetic mean per query and GitHub URL",
            "sourceRetrieval": "commit-pinned GitHub raw file plus annotated line range",
        },
        "statistics": {
            "annotationRows": len(python_rows),
            "aggregatedPairs": len(pair_scores),
            "sourceFilesRequested": len(file_sources),
            "sourceFilesFetched": len(fetched),
            "sourceFileFailures": len(failures),
            "snippetFailures": len(snippet_failures),
            "chunks": len(chunks),
            "queries": len(queries),
            "devQueries": sum(item["split"] == "dev" for item in queries),
            "testQueries": sum(item["split"] == "test" for item in queries),
            "unavailableJudgmentPairs": unavailable_pairs,
            "droppedQueries": len(dropped_queries),
        },
        "failures": sorted(failures, key=lambda item: item["key"]),
        "snippetFailures": sorted(snippet_failures, key=lambda item: item["url"]),
        "droppedQueries": dropped_queries,
        "chunks": sorted(chunks, key=lambda item: item["id"]),
        "queries": queries,
    }
    encoded = canonical_json(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output_file, mtime=0) as compressed:
            compressed.write(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "uncompressedBytes": len(encoded),
                "compressedBytes": args.output.stat().st_size,
                "payloadSha256": sha256_bytes(encoded),
                **payload["statistics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
