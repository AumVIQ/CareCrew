from data_analyze import client, GROQ_MODEL

def medical_context_icd(doc_analysis_text: str, language: str = "English") -> str:
    """
    Maps medical findings to ICD codes.
    Generates AI output in the selected language using Groq chat completions.
    """
    prompt = f"""Map the following medical findings to standard ICD codes:
{doc_analysis_text}

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
        return f"❌ ICD Mapping failed: {str(e)}"
