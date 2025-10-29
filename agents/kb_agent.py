import requests
from data_analyze import client, GROQ_MODEL, rag_lookup_kb

# MCP KB server endpoint
KB_MCP_URL = "http://127.0.0.1:8002/invoke_tool"


def _fetch_kb_via_mcp(query_text: str, language: str = "English", top_k: int = 4):
    """Queries the MCP KB server for relevant guideline passages."""
    try:
        payload = {
            "tool_name": "search_medical_guidelines",
            "arguments": {"query": query_text}
        }
        resp = requests.post(KB_MCP_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("result", {})
            if isinstance(data, dict) and "guideline_snippets" in data:
                return data["guideline_snippets"]
        return None
    except Exception:
        return None


def kb_agent(query_text, top_k=4, language="English"):
    """
    KB Agent: Fetches relevant passages from MCP Knowledge Base Server (preferred)
    or falls back to local RAG KB if MCP server is unavailable.
    Formats the passages into clear, easy-to-read bullet points in the selected language.
    """
    # --- Step 1: Try MCP server first ---
    hits = _fetch_kb_via_mcp(query_text, language, top_k=top_k)

    if not hits:
        # --- Step 2: Fallback to local RAG KB ---
        local_hits = rag_lookup_kb(query_text, top_k=top_k)
        if not local_hits:
            return "⚠️ No guideline passages found in KB (both MCP & local)."
        hits = [h.get("passage", "").strip() for h in local_hits if h.get("passage")]

    # --- Step 3: Clean and merge text snippets ---
    raw_passages = []
    for passage in hits:
        if len(passage) > 1000:
            passage = passage[:950].rsplit(" ", 1)[0] + "..."
        raw_passages.append(passage.strip())

    combined_text = "\n\n".join(raw_passages)

    # --- Step 4: Format the text into multilingual bullet points ---
    prompt = f"""
Summarize the following medical guideline passages into clear, evidence-based bullet points.
Each bullet point should be concise and informative.
Focus on diagnosis, treatment, and follow-up recommendations.
Generate the summary in {language}.

Passages:
{combined_text}
"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        formatted_output = resp.choices[0].message.content
        return formatted_output.strip()
    except Exception as e:
        return f"❌ Error formatting KB output: {str(e)}"
