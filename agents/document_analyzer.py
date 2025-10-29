from data_analyze import GROQ_MODEL, client, extract_text_from_pdf, file_to_base64
import os

def document_analyzer(file_paths, user_note=None, language="English") -> str:
    try:
        images = []
        texts = []

        # Collect all content
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[-1].lower()

            if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                image_b64 = file_to_base64(file_path)
                images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext[1:]};base64,{image_b64}"}
                })

            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append({"type": "text", "text": f.read()})

            elif ext == ".pdf":
                pdf_text = extract_text_from_pdf(file_path)
                texts.append({"type": "text", "text": pdf_text})

        reports = []

        # Function to run a batch safely
        def run_batch(batch_content, note=None):
            try:
                prompt = "Extract all lab values, symptoms, and abnormalities from the following medical reports. Provide concise bullet points."
                # Add user note if provided
                if note:
                    prompt += f"\nUser Note: {note}"
                # Add language instruction
                prompt += f"\nPlease generate the output in {language}."

                combined_input = [{"role": "user", "content": batch_content + [{"type": "text", "text": prompt}]}]

                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=combined_input,
                    temperature=0.2
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"⚠️ Batch failed: {str(e)}"

        # --- Step 1: Process images in batches of 5 (Groq hard limit) ---
        batch_size = 5
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size] + texts
            reports.append(run_batch(batch, user_note))

        # --- Step 2: If no images, just process texts ---
        if not images and texts:
            reports.append(run_batch(texts, user_note))

        # --- Step 3: If still no reports, fallback ---
        if not reports:
            return "⚠️ No valid content found to analyze."

        return "\n\n---\n\n".join(reports)

    except Exception as e:
        return f"❌ Document Analyzer failed: {str(e)}"
