from data_analyze import client, GROQ_MODEL

def reasoning_agent(icd_mapping_text: str, language: str = "English") -> str:
    """
    Performs clinical reasoning based on ICD-mapped data.
    Generates AI output in the selected language using Groq chat completions.
    """
    prompt = f"""Provide clinical reasoning and insights for the following ICD-mapped findings:
{icd_mapping_text}

Please generate the output in {language}.
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Clinical Reasoning failed: {str(e)}"
