from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from transformers import AutoTokenizer


@register_tool("search", allow_overwrite=True)
class SearchToolHandler(BaseTool):
    name = "search"

    def __init__(self, searcher, snippet_max_tokens: int = 512, k: int = 5):
        super().__init__()

        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k

        self.description = f"Performs a search on a knowledge source: supply a single 'query' string; the tool retrieves the top {self.k} most relevant results."

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen3/Qwen3-Embedding-0.6B")

    def _truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def _format_results(self, results: List[dict], max_tokens: int):
        formatted = []
        for r in results:
            passage_text = r["text"]
            title = ""

            if passage_text.startswith("---\ntitle:"):
                lines = passage_text.split("\n")
                if len(lines) > 1:
                    title = lines[1].replace("title:", "").strip().strip("\"")

            if not title:
                first_line = passage_text.split('\n')[0].strip()
                title = first_line[:50] + "..." if len(first_line) > 50 else first_line

            snippet = self._truncate(passage_text, max_tokens)

            formatted_result = f"DocID:{r['docid']}\n[{title}]\n{snippet}"
            formatted.append(formatted_result)

        return formatted


    def search_with_searcher(self, query: str, k: int = None):
        try:
            if k is None:
                k = self.k
            
            results = self.searcher.search(query, k)
            
            if not results:
                return f"No results found for '{query}'. Try with a more general query.", []
            
            docids = []
            for r in results:
                if "docid" in r:
                    docids.append(str(r["docid"]))

            formatted_results = self._format_results(results, self.snippet_max_tokens)

            content = f"A search for '{query}' found {len(formatted_results)} results:\n\n## Web Results\n" + "\n\n".join(formatted_results)
            
            return content, docids

        except Exception as e:
            return f"Search error for query '{query}': {str(e)}", []

    def call(self, params: Union[str, dict], **kwargs):
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field", None

        if not isinstance(query, str):
            if isinstance(query, list):
                query = query[0]
            else:
                return "[Search] Invalid request format: 'query' must be a string, not an array", None
        
        response, docids = self.search_with_searcher(query)
        
        return response, docids

@register_tool("get_document", allow_overwrite=True)
class GetDocumentToolHandler(BaseTool):
    name = "get_document"
    def __init__(self, searcher):
        super().__init__()
        self.searcher = searcher
        self.description = "Retrieve full document content based on provided docid(s)."
        self.document_max_tokens = 512
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen3/Qwen3-Embedding-0.6B")

    def _truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    def call(self, params: Union[str, dict], **kwargs):
        try:
            docid = params["docid"]
        except:
            return "[get_document] Invalid request format: Input must be a JSON object containing 'docid' field", None
    
        if isinstance(docid, list):
            docid = docid[0]

        results = []
        collected_docids = []

        try:
            content = self.searcher.get_document(docid)
            if not content:
                results.append(f"[Document not found] docid={docid}")
            else:
                truncted_content = self._truncate(content['text'], self.document_max_tokens)
                results.append(f"Document {docid}:\n{truncted_content}.")
                collected_docids.append(str(docid))
        except Exception as e:
            results.append(f"[Error retrieving {docid}]: {str(e)}")

        response_text = "\n\n".join(results)
        return response_text, collected_docids