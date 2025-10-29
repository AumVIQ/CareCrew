from crewai import Crew, Process, Task, Agent
from crewai.tools import BaseTool
import requests               
import json                   
from data_analyze import client, ensure_kb_index
from agents.document_analyzer import document_analyzer as doc_analyzer_logic
from agents.medical_context_agent import medical_context_icd as icd_logic
from agents.reasoning_agent import reasoning_agent as reasoning_logic 
from agents.kb_agent import kb_agent as kb_logic
from agents.treatment_planner_agent import treatment_planner_agent as planner_logic
from agents.advisory_agent import advisory_agent as advisory_logic
from agents.agent_definitions import ( 
    get_document_analyzer_agent, get_medical_context_agent, get_reasoning_agent, 
    get_kb_agent, get_treatment_planner_agent, get_advisory_agent
)
import groq
from yarl import URL



FDA_SERVER_URL = "http://127.0.0.1:8001/invoke_tool"
KB_SERVER_URL = "http://127.0.0.1:8002/invoke_tool"

class FdaCheckerTool(BaseTool):
    name: str = "check_drug_safety"
    description: str = "Check FDA safety information for a single drug. Input must be a single drug name string."

    def _run(self, drug_name: str) -> str:
        """Calls the FDA MCP server tool."""
        print(f"🤖 [Crew] Calling FDA Server for: {drug_name}")
        try:
            payload = {
                "tool_name": "check_drug_safety",
                "arguments": {"drug_name": drug_name}
            }
            response = requests.post(FDA_SERVER_URL, json=payload, timeout=10)
            response.raise_for_status() # Raise error for bad responses
            return json.dumps(response.json().get("result", "No result found"))
        except Exception as e:
            print(f"❌ [Crew] Failed to call FDA server: {str(e)}")
            return f"Error connecting to FDA server: {str(e)}"

class KbSearchTool(BaseTool):
    name: str = "search_medical_guidelines"
    description: str = "Search the Standard Treatment Guidelines (STG) knowledge base. Input must be a clinical query string."

    def _run(self, query: str) -> str:
        """Calls the KB MCP server tool."""
        print(f"🤖 [Crew] Calling KB Server for query: {query}")
        try:
            payload = {
                "tool_name": "search_medical_guidelines",
                "arguments": {"query": query, "top_k": 4} # Hardcode top_k=4 for simplicity
            }
            response = requests.post(KB_SERVER_URL, json=payload, timeout=10)
            response.raise_for_status() # Raise error for bad responses
            return json.dumps(response.json().get("result", "No result found"))
        except Exception as e:
            print(f"❌ [Crew] Failed to call KB server: {str(e)}")
            return f"Error connecting to KB server: {str(e)}"

# --- Instantiate the new tool classes ---
fda_checker_tool = FdaCheckerTool()
kb_search_tool = KbSearchTool()


def fix_groq_url(tool):
    """Ensure Groq models inside tools use string URLs, not URL objects."""
    if hasattr(tool, 'model') and isinstance(tool.model, groq.Groq):
        tool.model = groq.Groq(url=str(tool.model.url))
    return tool

def safe_task(callback, task_name):
    """Execute a task safely and log success/failure in terminal."""
    try:
        output = callback(None)
        print(f"✅ {task_name} succeeded.")
        return output
    except Exception as e:
        print(f"❌ {task_name} failed: {e}")
        return f"❌ Task failed: {task_name} | Error: {e}"

def run_medical_crew(file_paths: list, user_note: str = None, language: str = "English") -> str:
    # --- MCP tools ---


    # --- Agents ---
    try:
        document_analyzer_a = get_document_analyzer_agent()
        medical_context_agent = get_medical_context_agent()
        reasoning_agent_a = get_reasoning_agent() 
        kb_agent = get_kb_agent(tools=[kb_search_tool]) 
        treatment_planner_agent = get_treatment_planner_agent(tools=[fda_checker_tool]) 
        advisory_agent = get_advisory_agent()
    except Exception as e:
        return f"❌ Failed to initialize agents: {e}"

    # --- Execute tasks safely ---
    doc_analysis_output = safe_task(
        lambda _: doc_analyzer_logic(file_paths, user_note, language=language),
        "Document Analysis"
    )

    icd_mapping_output = safe_task(
        lambda _: icd_logic(doc_analysis_output, language=language),
        "ICD Mapping"
    )

    reasoning_output = safe_task(
        lambda _: reasoning_logic(icd_mapping_output, language=language),
        "Clinical Reasoning"
    )

    kb_lookup_output = safe_task(
        lambda _: kb_logic(reasoning_output, language=language),  # Use logic function
        "KB Lookup"
    )

    treatment_output = safe_task(
        lambda _: planner_logic(
            doc_analysis_output + icd_mapping_output + reasoning_output,
            kb_snippets=kb_lookup_output,
            language=language
        ),
        "Treatment Planning"
    )

    advisory_output = safe_task(
        lambda _: advisory_logic(treatment_output, language=language),
        "Patient Advisory"
    )

    # --- Final report ---
    final_report_parts = [
        f"### 📄 Document Analysis\n{doc_analysis_output}",
        f"### 🏷️ Medical Context (ICD)\n{icd_mapping_output}",
        f"### 🧠 Clinical Reasoning\n{reasoning_output}",
        f"### 🗂️ KB Guidelines\n{kb_lookup_output}",
        f"### 🩺 Treatment Plan\n{treatment_output}",
        f"### 💡 Patient Advisory\n{advisory_output}",
    ]

    return "\n\n---\n\n".join(final_report_parts)


if __name__ == '__main__':
    try:
        ensure_kb_index()
    except Exception as e:
        print(f"❌ Failed KB setup: {e}")
        
