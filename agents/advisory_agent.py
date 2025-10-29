from data_analyze import client, GROQ_MODEL

def advisory_agent(data: str, language: str = "English") -> str:
    """
    Converts clinical reasoning and treatment plan into patient-friendly advice.
    The output will be in the selected language.
    """
    try:
        prompt = f"""Rewrite the clinical reasoning and treatment plan below into simple, patient-friendly advice:
{data}

Please generate the output in {language}.
"""
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Advisory Agent failed: {str(e)}"
