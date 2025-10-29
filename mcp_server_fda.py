# mcp_server_fda.py - FDA MCP Server (Fixed for CrewAI)
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from fda_server_logic import _call_openfda_api, _call_openfda_api_batch

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(title="FDA MCP Server", version="1.0")

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

class FdaSingleDrugOutput(BaseModel):
    drug_name: str = Field(description="The name of the drug checked.")
    brand: str = Field(description="The brand name found, if available.")
    generic: str = Field(description="The generic name found, if available.")
    warnings: str = Field(description="FDA warnings or safety notes.")
    found: bool = Field(description="True if data found, False otherwise.")

class FdaBatchOutput(BaseModel):
    status: str = Field(description="Overall status of the batch operation.")
    count: int = Field(description="Total number of drugs processed.")
    results: List[FdaSingleDrugOutput] = Field(description="List of individual drug results.")

# ---------------------------------------------------
# Tool Functions
# ---------------------------------------------------
def check_drug_safety(drug_name: str) -> FdaSingleDrugOutput:
    """Check FDA safety information for a single drug."""
    if not drug_name or not drug_name.strip():
        return FdaSingleDrugOutput(
            drug_name="N/A",
            brand="N/A",
            generic="N/A",
            warnings="Invalid or empty drug name provided.",
            found=False
        )

    result = _call_openfda_api(drug_name)
    print(f"🩺 [FDA MCP] Processing single drug: {drug_name}")

    return FdaSingleDrugOutput(
        drug_name=drug_name,
        brand=result.get("brand", "N/A"),
        generic=result.get("generic", "N/A"),
        warnings=result.get("warnings", "No warnings available."),
        found=result.get("found", False)
    )

def check_multiple_drugs(drug_list: List[str]) -> FdaBatchOutput:
    """Check FDA safety information for multiple drugs."""
    print(f"\n🔍 [FDA MCP] Starting batch check for {len(drug_list)} drugs...\n")
    
    if not drug_list or not any(d.strip() for d in drug_list):
        return FdaBatchOutput(status="error", count=0, results=[])

    batch_result = _call_openfda_api_batch(drug_list)
    results = [
        FdaSingleDrugOutput(
            drug_name=e.get("drug_name", "N/A"),
            brand=e.get("brand", "N/A"),
            generic=e.get("generic", "N/A"),
            warnings=e.get("warnings", "No warnings available."),
            found=e.get("found", False)
        )
        for e in batch_result.get("results", [])
    ]

    print(f"\n✅ [FDA MCP] Batch check completed ({len(results)} drugs).\n")

    return FdaBatchOutput(
        status=batch_result.get("status", "success"),
        count=len(results),
        results=results
    )

# ---------------------------------------------------
# MCP Endpoints
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "FDA MCP Server is running",
        "version": "1.0",
        "available_tools": ["check_drug_safety", "check_multiple_drugs"]
    }

@app.get("/tools")
def list_tools():
    """List available MCP tools."""
    return {
        "tools": [
            {
                "name": "check_drug_safety",
                "description": "Check FDA safety information for a single drug",
                "parameters": {
                    "drug_name": {
                        "type": "string",
                        "description": "The exact brand or generic name of the drug to check"
                    }
                }
            },
            {
                "name": "check_multiple_drugs",
                "description": "Check FDA safety information for multiple drugs",
                "parameters": {
                    "drug_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of drug names to check"
                    }
                }
            }
        ]
    }

@app.post("/invoke_tool")
def invoke_tool(invocation: ToolInvocation):
    """Invoke an MCP tool."""
    print(f"🔧 [FDA MCP] Invoking tool: {invocation.tool_name}")
    print(f"📥 [FDA MCP] Arguments: {invocation.arguments}")
    
    try:
        if invocation.tool_name == "check_drug_safety":
            drug_name = invocation.arguments.get("drug_name", "")
            result = check_drug_safety(drug_name)
            return {"success": True, "result": result.model_dump()}
        
        elif invocation.tool_name == "check_multiple_drugs":
            drug_list = invocation.arguments.get("drug_list", [])
            result = check_multiple_drugs(drug_list)
            return {"success": True, "result": result.model_dump()}
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown tool: {invocation.tool_name}"
            )
    
    except Exception as e:
        print(f"❌ [FDA MCP] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------
# Server Runner
# ---------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting FDA MCP Server on port 8001")
    print("📡 MCP endpoint: http://127.0.0.1:8001/invoke_tool")  # <-- FIXED
    uvicorn.run(app, host="127.0.0.1", port=8001)