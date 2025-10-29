# data_analyze.py - CORRECT AND COMPLETE VERSION

import os
import re
import pickle
import base64
import fitz
import requests
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Groq Setup
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# Supported Languages
# -------------------------
SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Kannada"]

# -------------------------
# Knowledge Base Setup
# -------------------------
STG_PDF = "standard-treatment-guidelines.pdf"
KB_INDEX_PICKLE = "kb_index.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
KB_EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
RAG_EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
kb_index_data = None

# -------------------------
# FDA Configuration
# -------------------------
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
FDA_MCP_URL = "http://127.0.0.1:8001/invoke_tool"
KB_MCP_URL = "http://127.0.0.1:8002/invoke_tool"


# -------------------------
# Utility Functions
# -------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def file_to_base64(file_path: str):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# -------------------------
# FDA Logic (CLIENT-SIDE)
# -------------------------
def get_openfda_warnings(drug_name: str) -> Dict[str, str]:
    """
    Fetches FDA safety warnings for a single drug name.
    1️⃣ Tries local MCP (port 8001)
    2️⃣ Falls back to OpenFDA REST API
    """
    # --- Step 1: Try Local MCP Server ---
    try:
        payload = {
            "tool_name": "check_drug_safety",
            "arguments": {"drug_name": drug_name}
        }
        
        print(f"\n--- 📞 ATTEMPTING TO CALL LOCAL FDA SERVER FOR: {drug_name} ---\n")
        
        resp = requests.post(FDA_MCP_URL, json=payload, timeout=8)
        
        if resp.status_code == 200:
            data = resp.json().get("result", {})
            if isinstance(data, dict):
                print(f"✅ [MCP FDA Check] {drug_name} → Response OK")
                return {
                    "drug_name": drug_name,
                    "brand": data.get("brand", "N/A"),
                    "generic": data.get("generic", "N/A"),
                    "warnings": data.get("warnings", "No warnings available."), # Corrected key
                    "found": True
                }
        else:
            # If the status code was NOT 200, raise an error
            print(f"--- ‼️ LOCAL SERVER FAILED WITH STATUS CODE: {resp.status_code} ---")
            raise Exception(f"Local FDA server returned status {resp.status_code}: {resp.text}")

    except Exception as e:
        # This will now print the explicit error from above
        print(f"⚠️ MCP connection failed for {drug_name}: {e}")
        # DO NOT raise e here, we want it to fall through to the fallback

    # --- Step 2: Fallback to OpenFDA REST API ---
    print(f"--- 🌐 FALLING BACK TO PUBLIC INTERNET API FOR: {drug_name} ---")
    try:
        for key in ["brand_name", "generic_name"]:
            params = {"search": f"openfda.{key}:{drug_name}", "limit": 1}
            url = f"{OPENFDA_BASE}?search=openfda.{key}:{drug_name}&limit=1"
            print(f"🌐 [FDA API Called]: {url}")
            resp = requests.get(OPENFDA_BASE, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("results", [])
                if data:
                    entry = data[0]
                    brand = entry.get("openfda", {}).get("brand_name", ["N/A"])[0]
                    generic = entry.get("openfda", {}).get("generic_name", ["N/A"])[0]
                    warnings = entry.get("warnings", ["No warnings available."])[0]
                    print(f"✅ [FDA Data Found] {drug_name} → {brand}/{generic}")
                    return {
                        "drug_name": drug_name,
                        "brand": brand,
                        "generic": generic,
                        "warnings": warnings,
                        "found": True
                    }
    except Exception as e_fallback:
        print(f"⚠️ OpenFDA API fallback error for {drug_name}: {e_fallback}")

    # --- Step 3: Return fallback result ---
    print(f"❌ [FDA Not Found] {drug_name}")
    return {
        "drug_name": drug_name,
        "brand": "N/A",
        "generic": "N/A",
        "warnings": "No data found.",
        "found": False
    }


def get_openfda_warnings_batch(medicine_list: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Performs FDA safety check for all medicines found in a list.
    Returns a combined JSON result for all drugs.
    """
    results = []
    print("\n🔍 Starting FDA Checkup for All Detected Medicines...\n")
    for drug in medicine_list:
        clean_name = drug.strip()
        if not clean_name:
            continue
        result = get_openfda_warnings(clean_name)
        results.append(result)

    print("\n✅ FDA Checkup Completed for All Medicines.\n")
    return {
        "status": "success" if results else "no_data",
        "count": len(results),
        "results": results
    }

# -------------------------
# KB Index Management
# -------------------------
def build_kb_index(pdf_path: str = STG_PDF, out_pickle: str = KB_INDEX_PICKLE):
    print(f"Building KB (STG) index from PDF: {pdf_path}...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Cannot build KB index. PDF file not found at: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    text = re.sub(r"\n{2,}", "\n", text)
    passages = chunk_text(text, chunk_size=300, overlap=50)
    embeddings = KB_EMBEDDER.encode(passages, convert_to_numpy=True, show_progress_bar=True)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    with open(out_pickle, "wb") as f:
        pickle.dump({"index": index, "passages": passages}, f)
    print(f"✅ Built KB index with {len(passages)} passages and saved to {out_pickle}")
    return {"index": index, "passages": passages}

def load_kb_index(pickle_path: str = KB_INDEX_PICKLE):
    if not os.path.exists(pickle_path):
        return None
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def ensure_kb_index():
    global kb_index_data
    if kb_index_data is None:
        data = load_kb_index(KB_INDEX_PICKLE)
        if data is None:
            if os.path.exists(STG_PDF):
                data = build_kb_index(STG_PDF, KB_INDEX_PICKLE)
            else:
                print(f"⚠️ STG PDF ({STG_PDF}) not found. RAG will not work.")
                return None
        kb_index_data = data
    return kb_index_data

def rag_lookup_kb(query: str, top_k: int = 4) -> List[Dict[str, str]]:
    kb_data = ensure_kb_index()
    if kb_data is None:
        return []
    emb = RAG_EMBEDDER.encode([query], convert_to_numpy=True)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    D, I = kb_data["index"].search(emb, top_k)
    results = []
    for idx in I[0]:
        if idx >= 0 and idx < len(kb_data["passages"]):
            results.append({"passage": kb_data["passages"][idx], "score": float(D[0][list(I[0]).index(idx)])})
    return results

# -------------------------
# Language Helpers
# -------------------------
def build_ai_prompt(base_text: str, language: str = "English") -> str:
    """Construct a multilingual Groq prompt."""
    if language not in SUPPORTED_LANGUAGES:
        language = "English"
    return f"{base_text}\nPlease generate the output in {language}."

def groq_query(prompt_text: str) -> str:
    """Run a Groq query using the Groq client."""
    response = client.generate(prompt_text, model=GROQ_MODEL)
    return getattr(response, "output_text", str(response))

def analyze_data_with_language(data: str, language: str = "English") -> str:
    """Example multilingual Groq analysis."""
    prompt = build_ai_prompt(f"Analyze the following data:\n{data}", language)
    return groq_query(prompt)