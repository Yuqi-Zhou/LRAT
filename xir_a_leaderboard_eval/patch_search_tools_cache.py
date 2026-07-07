#!/usr/bin/env python3
"""Patch BrowseComp-Plus searcher/tools.py with lightweight caches."""

from __future__ import annotations

import argparse
from pathlib import Path


def patch(path: Path) -> bool:
    src = path.read_text()
    if "search_cache: Dict[str, List[Dict[str, Any]]]" in src:
        return False

    backup = path.with_suffix(path.suffix + ".bak_cache")
    if not backup.exists():
        backup.write_text(src)

    src = src.replace(
        "    tokenizer = None\n    if snippet_max_tokens and snippet_max_tokens > 0:",
        "    tokenizer = None\n"
        "    search_cache: Dict[str, List[Dict[str, Any]]] = {}\n"
        "    snippet_cache: Dict[str, str] = {}\n"
        "    document_cache: Dict[str, Optional[Dict[str, Any]]] = {}\n\n"
        "    if snippet_max_tokens and snippet_max_tokens > 0:",
    )

    old_search = '''    def search(
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Search the index and return top-k hits
        Args:
            query: Search query string
        Returns:
            List of search results with docid, score, text
        """
        candidates = searcher.search(query, k)

        if snippet_max_tokens and snippet_max_tokens > 0 and tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > snippet_max_tokens:
                    truncated_tokens = tokens[:snippet_max_tokens]
                    cand["snippet"] = tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return results
'''
    new_search = '''    def search(
        query: str,
    ) -> List[Dict[str, Any]]:
        """Search the index and return top-k hits with exact-query caching."""
        normalized_query = str(query or "").strip()
        if normalized_query in search_cache:
            return search_cache[normalized_query]

        candidates = searcher.search(normalized_query, k)

        if snippet_max_tokens and snippet_max_tokens > 0 and tokenizer:
            for cand in candidates:
                docid = str(cand["docid"])
                if docid in snippet_cache:
                    cand["snippet"] = snippet_cache[docid]
                    continue
                text = cand["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > snippet_max_tokens:
                    truncated_tokens = tokens[:snippet_max_tokens]
                    cand["snippet"] = tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
                snippet_cache[docid] = cand["snippet"]
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        search_cache[normalized_query] = results
        return results
'''
    if old_search not in src:
        raise RuntimeError("Could not find original search() block to patch")
    src = src.replace(old_search, new_search)

    old_doc = '''        def get_document(docid: str) -> Optional[Dict[str, Any]]:
            """
            Retrieve the full text of a document by its ID from the search index.

            Args:
                docid: Document ID to retrieve

            Returns:
                Document with full text, or None if not found
            """
            return searcher.get_document(docid)
'''
    new_doc = '''        def get_document(docid: str) -> Optional[Dict[str, Any]]:
            """
            Retrieve the full text of a document by its ID from the search index.

            Args:
                docid: Document ID to retrieve

            Returns:
                Document with full text, or None if not found
            """
            docid = str(docid)
            if docid not in document_cache:
                document_cache[docid] = searcher.get_document(docid)
            return document_cache[docid]
'''
    if old_doc not in src:
        raise RuntimeError("Could not find original get_document() block to patch")
    src = src.replace(old_doc, new_doc)

    path.write_text(src)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tools_py", type=Path)
    args = parser.parse_args()
    changed = patch(args.tools_py)
    print(f"{args.tools_py}: {'patched' if changed else 'already patched'}")


if __name__ == "__main__":
    main()
