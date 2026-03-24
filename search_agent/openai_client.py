import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompts import format_query
from searcher.searchers import SearcherType


logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_api_key() -> str:
    return os.getenv("API_KEYS") or os.getenv("API_KEY") or ""


def get_base_url() -> str:
    return os.getenv("URL") or os.getenv("BASE_URL") or ""


class SearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.document_max_tokens = 512
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen3/Qwen3-0.6B")
    
    def _truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def get_tool_definitions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_query": {
                                "type": "string",
                                "description": "Query to search the local knowledge base for relevant information",
                            }
                        },
                        "required": ["user_query"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                    },
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["user_query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)
        
        for cand in candidates:
            text = cand.get("text", "")
            cand["snippet"] = self._truncate(text, self.snippet_max_tokens)

        results = []
        for cand in candidates:
            entry = {"docid": cand["docid"], "snippet": cand["snippet"]}
            if cand.get("score") is not None:
                entry["score"] = cand["score"]
            results.append(entry)

        return json.dumps(results, indent=2, ensure_ascii=False)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})

        if "text" in result:
            result['text'] = self._truncate(result['text'], self.document_max_tokens)
        return json.dumps(result, indent=2, ensure_ascii=False)


def handle_conversation(args, tool_handler, query_text, qid=None):
    tool_choice = 'auto'
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": format_query(query_text, args.query_template)}
    ]
    tools = tool_handler.get_tool_definitions()
    tool_usage_counts = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for i in range(args.max_iterations):
        is_last_round = (i == args.max_iterations - 2)
        response = None
        for i in range(5):
            try:
                client = OpenAI(base_url=get_base_url(), api_key=get_api_key())
                if 'minimax' in args.model.lower():
                    extra_body={"reasoning_split": True}
                elif 'glm' in args.model.lower():
                    extra_body={
                        "thinking":{
                        "type":"enabled",
                        "clear_thinking": False
                    }}
                else:
                    extra_body={}
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=args.temperature if args.temperature is not None else 1.0,
                    max_tokens=args.max_tokens,
                    extra_body=extra_body
                )
                break
            except Exception as e:
                if i < 5:
                    logger.warning("Chat completion attempt %s failed: %s", i + 1, e)
                    time.sleep(5)

        if response is None:
            break

        res_msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(res_msg)

        if is_last_round:
            messages.append({
                "role": "user", 
                "content": "Retrieval complete. Please provide your final answer based on the above info."
            })
            tool_choice = 'none'
            continue 

        if res_msg.tool_calls and len(res_msg.tool_calls) > 0:

            try:
                for tool_call in res_msg.tool_calls:
                    t_name = tool_call.function.name
                    t_id = tool_call.id
                    t_args = json.loads(tool_call.function.arguments)

                    tool_usage_counts[t_name] = tool_usage_counts.get(t_name, 0) + 1
                    
                    result_output = tool_handler.execute_tool(t_name, t_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": t_id,
                        "name": t_name,
                        "content": result_output
                    })
            except:
                messages.append({
                    "role": "tool",
                    "content": 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                })
                    
        if finish_reason == "stop":
            break

    _persist_response(
        args.output_dir, args.model, messages, tool_usage_counts, qid, total_usage
    )
    return messages[-1]

def _persist_response(out_dir, model_name, messages, tool_counts, query_id, total_usage):
    import os
    import json
    import datetime
    import re

    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        
        safe_qid = re.sub(r'[^\w\-_]', '_', str(query_id)) if query_id else "single"
        final_filename = os.path.join(out_dir, f"run_{safe_qid}_{ts}.json")

        T_START = "<think>"
        T_END = "</think>"
        think_pattern = rf"<{T_START}>(.*?)<{T_END}>"

        normalized_results = []
        
        for msg in messages:
            if hasattr(msg, 'model_dump'):
                m = msg.model_dump()
            elif isinstance(msg, dict):
                m = msg
            else:
                m = {"role": "unknown", "content": str(msg)}

            role = m.get("role")
            content = m.get("content") or ""

            if role == "assistant":
                found_think = re.search(think_pattern, content, re.DOTALL)
                if found_think:
                    normalized_results.append({
                        "type": "reasoning",
                        "output": found_think.group(1).strip()
                    })
                    content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()

                if content:
                    normalized_results.append({"type": "output_text", "output": content})

                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        raw_args = tc["function"].get("arguments", "{}")
                        if isinstance(raw_args, dict):
                            parsed_args = raw_args
                        else:
                            try:
                                parsed_args = json.loads(raw_args)
                            except:
                                parsed_args = {"raw_string": raw_args, "error": "json_parse_failed"}

                        normalized_results.append({
                            "type": "tool_call",
                            "id": tc.get("id"),
                            "tool_name": tc["function"].get("name"),
                            "arguments": parsed_args,
                            "output": None 
                        })

            elif role == "tool":
                res_id = m.get("tool_call_id")
                res_content = m.get("content")
                if isinstance(res_content, str) and len(res_content) > 200000:
                    res_content = res_content[:200000] + "...[truncated]"
                
                for entry in normalized_results:
                    if entry.get("type") == "tool_call" and entry.get("id") == res_id:
                        entry["output"] = res_content

        from utils import extract_retrieved_docids_from_result
        docids = extract_retrieved_docids_from_result(normalized_results)

        record = {
            "metadata": {
                "model": model_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "usage": total_usage
            },
            "query_id": query_id,
            "tool_call_counts": tool_counts,
            "retrieved_docids": docids,
            "result": normalized_results
        }

        with open(final_filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.exception("Failed to persist response: %s", e)

def main():
    parser = argparse.ArgumentParser(description="MiniMax-M2 DeepSearch Client")
    parser.add_argument("--query", default="queries.tsv", help="Query text or TSV path")
    parser.add_argument("--model", default="MiniMax-M2.1", help="MiniMax Model Name")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", default="runs/minimax_m2")
    parser.add_argument("--system", default="You are a helpful assistant with search tools.")
    parser.add_argument("--query-template", default="QUERY_TEMPLATE_NO_GET_DOCUMENT")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max loops for tool calling")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--searcher-type", required=True, choices=SearcherType.get_choices())
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--get-document", action="store_true")

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)
    args = parser.parse_args()

    searcher = searcher_class(args)
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    if args.query.endswith(".tsv"):
        queries = []
        with open(args.query, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    queries.append((row[0].strip(), row[1].strip()))
        
        processed_ids = set()
        out_dir = Path(args.output_dir).expanduser().resolve()
        
        if out_dir.exists():
            for json_path in out_dir.glob("run_*.json"):
                try:
                    with json_path.open("r", encoding="utf-8") as jf:
                        meta = json.load(jf)
                        qid_saved = meta.get("query_id")
                        if qid_saved:
                            processed_ids.add(str(qid_saved))
                except Exception:
                    continue
        
        remaining_queries = [q for q in queries if q[0] not in processed_ids]
        
        logger.info("Dataset total: %s", len(queries))
        logger.info("Already processed: %s", len(processed_ids))
        logger.info("Remaining to run: %s", len(remaining_queries))

        if not remaining_queries:
            logger.info("All queries in the TSV have been processed. Exiting.")
            return

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                executor.submit(handle_conversation, args, tool_handler, qtext, qid): qid 
                for qid, qtext in remaining_queries
            }

            with tqdm(total=len(remaining_queries), desc="Processing Queries") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Query %s failed: %s", qid, e)
                    pbar.update(1)
    else:
        logger.info("Processing single query")
        handle_conversation(args, tool_handler, args.query, None)
                    
if __name__ == "__main__":
    load_dotenv()
    main()
