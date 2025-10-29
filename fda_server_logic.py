# fda_server_logic.py

import requests
from typing import List, Dict, Any

# This is the external, public FDA API
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

def _call_openfda_api(drug_name: str) -> Dict[str, any]:
    """Directly calls the external OpenFDA API."""
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
                    print(f"✅ [FDA API] Found: {drug_name} -> {brand}/{generic}")
                    return {
                        "drug_name": drug_name,
                        "brand": brand,
                        "generic": generic,
                        "warnings": warnings,
                        "found": True
                    }
    except Exception as e:
        print(f"⚠️ OpenFDA API error for {drug_name}: {e}")

    print(f"❌ [FDA API] Not Found: {drug_name}")
    return {
        "drug_name": drug_name,
        "brand": "N/A",
        "generic": "N/A",
        "warnings": "No data found.",
        "found": False
    }

def _call_openfda_api_batch(medicine_list: List[str]) -> Dict[str, any]:
    """Directly calls external OpenFDA API for a batch."""
    results = []
    print("\n🔍 Starting BATCH FDA Checkup (Direct API)...\n")
    for drug in medicine_list:
        clean_name = drug.strip()
        if not clean_name:
            continue
        result = _call_openfda_api(clean_name) 
        results.append(result)
    
    print("\n✅ Direct API Batch Checkup Completed.\n")
    return {
        "status": "success" if results else "no_data",
        "count": len(results),
        "results": results
    }