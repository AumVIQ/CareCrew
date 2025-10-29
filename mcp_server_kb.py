# mcp_server_kb.py - Knowledge Base MCP Server (Fixed for CrewAI)


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from data_analyze import rag_lookup_kb

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(title="Knowledge Base MCP Server", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Data Models
# ---------------------------------------------------
class ToolInvocation(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class KBLookupOutput(BaseModel):
    guideline_snippets: List[str] = Field(description="Relevant passages from medical guidelines")
    query_used: str = Field(description="The clinical query used for retrieval")

# ---------------------------------------------------
# Tool Functions
# ---------------------------------------------------
def search_medical_guidelines(query: str, top_k: int = 4) -> KBLookupOutput:
    """Search medical guidelines knowledge base."""
    print(f"🔍 [KB MCP] Searching for: {query}")
    
    try:
        hits = rag_lookup_kb(query, top_k=top_k)
        snippets = [h['passage'] for h in hits]
        
        print(f"✅ [KB MCP] Found {len(snippets)} relevant guidelines")
        
        return KBLookupOutput(
            guideline_snippets=snippets,
            query_used=query
        )
    
    except Exception as e:
        print(f"❌ [KB MCP] Search error: {str(e)}")
        return KBLookupOutput(
            guideline_snippets=[f"Error during search: {str(e)}"],
            query_used=query
        )

# ---------------------------------------------------
# MCP Endpoints
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Knowledge Base MCP Server is running",
        "version": "1.0",
        "available_tools": ["search_medical_guidelines"]
    }

@app.get("/tools")
def list_tools():
    """List available MCP tools."""
    return {
        "tools": [
            {
                "name": "search_medical_guidelines",
                "description": "Search the Standard Treatment Guidelines (STG) knowledge base",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "The clinical query to search"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 4)",
                        "default": 4
                    }
                }
            }
        ]
    }

@app.post("/invoke_tool")
def invoke_tool(invocation: ToolInvocation):
    """Invoke an MCP tool."""
    print(f"🔧 [KB MCP] Invoking tool: {invocation.tool_name}")
    print(f"📥 [KB MCP] Arguments: {invocation.arguments}")
    
    try:
        if invocation.tool_name == "search_medical_guidelines":
            query = invocation.arguments.get("query", "")
            top_k = invocation.arguments.get("top_k", 4)
            
            result = search_medical_guidelines(query, top_k)
            return {"success": True, "result": result.model_dump()}
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tool: {invocation.tool_name}"
            )
    
    except Exception as e:
        print(f"❌ [KB MCP] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------
# Server Runner
# ---------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Knowledge Base MCP Server on port 8002")
    print("📡 MCP endpoint: http://127.0.0.1:8002/invoke_tool")  # <-- FIXED
    uvicorn.run(app, host="127.0.0.1", port=8002)