# agents/crew_tasks.py

from agents.document_analyzer import document_analyzer
from agents.medical_context_agent import medical_context_icd
from agents.reasoning_agent import reasoning_agent
from agents.kb_agent import kb_agent
from agents.treatment_planner_agent import treatment_planner_agent
from agents.advisory_agent import advisory_agent

def sequential_executor(file_paths, user_note=None):
    # Document Analysis
    doc_report = document_analyzer(file_paths, user_note)
    
    # Medical Context
    icd_report = medical_context_icd(doc_report)
    
    # Reasoning
    reasoning = reasoning_agent(doc_report + "\n" + icd_report)
    
    # Knowledge Base Lookup
    kb_snippets = kb_agent(doc_report + "\n" + icd_report + "\n" + reasoning, top_k=4)
    
    # Treatment Planning
    treatment = treatment_planner_agent(doc_report + "\n" + icd_report + "\n" + reasoning, kb_snippets=kb_snippets)
    
    # Advisory
    advisory = advisory_agent(doc_report + "\n" + icd_report + "\n" + reasoning + "\n" + treatment)
    
    return {
        "doc_report": doc_report,
        "icd_report": icd_report,
        "reasoning": reasoning,
        "kb_snippets": kb_snippets,
        "treatment": treatment,
        "advisory": advisory
    }
