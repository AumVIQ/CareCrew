import streamlit as st # type: ignore
import os
import tempfile
from datetime import datetime
from crew_orchestrator import run_medical_crew  # Correct import

# --- Streamlit App Config ---
st.set_page_config(
    page_title="🩺 Smart Healthcare Diagnostic Agent",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
.main-header { font-size: 2.5rem; font-weight: 700; color: #2c3e50; text-align: center; margin-bottom: 1rem; }
.sub-header { font-size: 1.2rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem; }
.stButton>button { background-color: #3498db; color: white; font-weight: bold; border-radius: 5px; padding: 0.5rem 1rem; border: none; transition: all 0.3s ease; }
.stButton>button:hover { background-color: #2980b9; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
.upload-section { background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border: 2px dashed #3498db; margin: 1rem 0; }
.result-card { background-color: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 5px solid #3498db; }
.chat-message { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
.user-message { background-color: #e8f4fc; border-left: 3px solid #3498db; }
.bot-message { background-color: #f0f0f0; border-left: 3px solid #2ecc71; }
.processing { display: inline-block; padding: 0.5rem 1rem; background-color: #fff3cd; color: #856404; border-radius: 20px; font-size: 0.9rem; margin: 0.5rem 0; }
.success { display: inline-block; padding: 0.5rem 1rem; background-color: #d4edda; color: #155724; border-radius: 20px; font-size: 0.9rem; margin: 0.5rem 0; }
.warning-box { background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 5px solid #ffc107; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("ℹ️ About This App")
    st.markdown("""
**Smart Healthcare Diagnostic Agent**

Upload medical documents for AI-powered analysis:
- 📄 PDF reports & prescriptions  
- 🖼️ Medical images (X-rays, scans)  
- 📊 Lab results & vitals data  

Powered by Groq LLM + RAG (for KB agent).
""")
    st.markdown("---")
    st.subheader("⚠️ Medical Disclaimer")
    st.markdown("""
This tool provides informational support only.  
Always consult with a qualified healthcare professional for medical diagnosis and treatment.
""")

# --- Main Header ---
st.markdown('<h1 class="main-header">🩺 Smart Healthcare Diagnostic Agent</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="warning-box">
⚠️ <strong>Important Medical Disclaimer:</strong> This AI tool is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2 = st.tabs(["📋 Upload & Analyze", "💬 Chat History"])

with tab1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    # --- Language Selection ---
    st.markdown("### 🌐 Choose Output Language")
    selected_language = st.selectbox(
        "Select the language for AI-generated reports:",
        ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Kannada"],
        index=0  # default to English
    )

    st.markdown("### 📁 Upload Medical Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Upload one or more files."
    )
    
    st.markdown("### 📝 Additional Information (Optional)")
    user_description = st.text_area(
        "Describe symptoms, concerns, or context",
        placeholder="E.g., Patient has been experiencing fatigue and dizziness for 2 weeks...",
        height=100,
        max_chars=500
    )
    st.caption(f"Characters: {len(user_description)}/500")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        if st.button("🚀 Analyze Documents", use_container_width=True, type="primary"):
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Save uploaded files temporarily
            file_paths, file_names = [], []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.read())
                    file_paths.append(tmp_file.name)
                    file_names.append(file.name)

            user_message = f"Analyzed {len(file_names)} file(s): {', '.join(file_names)}"
            if user_description:
                user_message += f"\n\nWith additional context: {user_description}"
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            status_placeholder = st.empty()
            status_placeholder.markdown(f'<div class="processing">Processing {len(file_names)} files with CrewAI...</div>', unsafe_allow_html=True)

            try:
                # Pass language parameter to backend
                analysis = run_medical_crew(file_paths, user_description, language=selected_language)

                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"### 🧾 Combined Analysis Report ({selected_language})")
                    st.markdown(f'<div class="file-info">Files analyzed: {", ".join(file_names)}</div>', unsafe_allow_html=True)
                    st.markdown(analysis)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": analysis,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                status_placeholder.markdown('<div class="success">✅ Analysis Complete!</div>', unsafe_allow_html=True)

                full_report = f"""Healthcare Diagnostic Analysis Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Language: {selected_language}

{analysis}

---
Disclaimer: This report is for informational purposes only.
"""
                st.download_button(
                    "📥 Download Full Report (TXT)",
                    data=full_report,
                    file_name=f"healthcare_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            finally:
                for path in file_paths:
                    if os.path.exists(path):
                        os.unlink(path)

with tab2:
    st.markdown("### 💬 Analysis History")
    if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
        st.info("No history yet.")
    else:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            timestamp = chat.get("timestamp", "Unknown time")
            if chat["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>👤 You</strong> <small>({timestamp})</small><br>{chat["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><strong>🤖 AI Assistant</strong> <small>({timestamp})</small><br>{chat["content"]}</div>', unsafe_allow_html=True)
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.chat_history = []
            st.experimental_rerun()

st.markdown("---")
