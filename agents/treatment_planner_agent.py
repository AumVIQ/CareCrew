import re
from data_analyze import client, GROQ_MODEL, get_openfda_warnings_batch

# --------------------------
# Step 1️⃣ - Smart Drug Extraction (Groq-based)
# --------------------------
def extract_drugs_with_groq(text: str):
    """
    Uses Groq LLM to intelligently identify medicine names from text.
    Always returns English drug names for OpenFDA compatibility.
    """
    try:
        prompt = f"""
You are a medical NLP assistant.
Extract ONLY medicine or drug names mentioned in the following text.
Ignore dosage, form (mg, tablet, syrup, injection), and instructions.
Return a comma-separated list of drug names in ENGLISH only.

Text:
{text}
"""
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        output = response.choices[0].message.content.strip()
        drugs = [d.strip().lower() for d in re.split(r"[,\n]+", output) if d.strip()]
        print(f"💊 [Groq Extracted Medicines]: {', '.join(drugs) if drugs else 'None'}")
        return drugs

    except Exception as e:
        print(f"❌ Groq extraction failed: {e}")
        return []


# --------------------------
# Step 2️⃣ - Treatment Planner Agent
# --------------------------
def treatment_planner_agent(data: str, kb_snippets: str = None, language: str = "English") -> str:
    """
    Generates a clear treatment plan using Groq,
    extracts drugs (via Groq),
    performs batch FDA safety checks,
    and appends multilingual report.
    """
    try:
        # --- Step 1: Generate Treatment Plan ---
        plan_prompt = f"""
You are a multilingual medical assistant.
Based on the patient's findings and KB snippets, create a short, structured treatment plan.
Use bullet points and clear recommendations.
Generate the plan in {language}. Do NOT include FDA warnings yet.

INPUT:
{data}

KB SNIPPETS:
{kb_snippets if kb_snippets else 'No KB snippets provided.'}
"""
        plan_resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": plan_prompt}],
            temperature=0.3
        )
        plan = plan_resp.choices[0].message.content.strip()

        # --- Step 2: Extract Medicines using Groq ---
        drugs = extract_drugs_with_groq(plan + " " + data)
        if not drugs:
            return plan + "\n\n💊 No medicines detected for FDA verification."

        # --- Step 3: Perform FDA Batch Check ---
        print("\n🔍 Performing batch FDA check for extracted medicines...\n")
        fda_batch = get_openfda_warnings_batch(drugs)
        fda_results = fda_batch.get("results", [])
        if not fda_results:
            return plan + "\n\n⚠️ FDA safety check failed or returned no data."

        # --- Step 4: Build FDA Safety Summary ---
        safety_notes = []
        for i, entry in enumerate(fda_results, start=1):
            drug = entry.get("drug_name", "N/A").title()
            brand = entry.get("brand", "N/A")
            generic = entry.get("generic", "N/A")
            warning = entry.get("warnings", "No safety info available.")
            note = f"**{i}. {drug}** ({brand}/{generic})\n   - ⚕️ {warning}"
            safety_notes.append(note)

        fda_summary = "\n\n---\n\n💊 **FDA Safety Summary:**\n" + "\n\n".join(safety_notes)
        final_output = plan + fda_summary

        # --- Step 5: Translate final output (if needed) ---
        if language.lower() != "english":
            translate_prompt = f"""
Translate the following medical report into {language}.
Keep headings, formatting, and emojis intact.

{final_output}
"""
            trans_resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": translate_prompt}],
                temperature=0.2
            )
            translated = trans_resp.choices[0].message.content.strip()
            print("\n🌍 [Translation Completed]")
            return translated

        # --- Step 6: Return English Report ---
        print("\n✅ [Treatment Plan + FDA Summary Generated Successfully]")
        return final_output

    except Exception as e:
        return f"❌ Treatment Planner Agent failed: {str(e)}"
