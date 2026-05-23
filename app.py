"""
app.py  --

Stramlit application for AI-powered resume analaysis using :
-Langchain (RAG pipeline) 
-huggingFace Interface API (FREE LLM)
-FAISS (vector similarity search) 
-senetence-transformers (local embedding model, no API calls)

Run:
streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from resume_parser import extract_text_from_resume
from rag_pipeline import ResumeRAGPipeline, AVAILABLE_MODELS

load_dotenv()  # Load environment variables from .env file

# ------------------------
# Page configuration
# ------------------------

st.set_page_config(
    page_title="Resume Checker AI",
    page_icon = "",
    layout = "wide"
)

#------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>
   /* Title */
   h1 {text-align: center; color: #1a73e8;}
            
   /* Score badge */
   .score-badge {
       font-size: 2rem; font-weight: bold;
      background: linear-gradient(135deg, #1a73e8, #0d47a1);
      color: white; border-radius: 12px;
      padding: 12px 24px; display: inline-block;
      margin-bottom: 12px;      
   } 
   /* Section cards */
   .card{
            background: #f8f9fa;
            border-left: 4px solid #1a73e8;
            border-radius: 8px;
            padding: 16px; margin-bottom: 16px;
            }                  
   /*Strength card */
          .card-green {border-left-color: #2e7d32;}
   /*Weakness card */
          .card-red {border-left-color: #c62828;}
   /*Suggestion card */
          .card-orange {border-left-color: #e65100;}
   /* ATS  card */  
          .card-purple {border-left-color: #6a1b9a;}
</style>
""", unsafe_allow_html=True)


# ------------------------
# Header
# ------------------------

st.markdown("<h1>Resume Checker AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #555; font-size:1.1rem;'>Upload your resume (PDF, DOCX, or TXT) and get instant AI rating , strenghts, gaps, and improvement tips - 100% FREE"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# -----------------------
# Sidebar - Configuration
# -----------------------

with st.sidebar:
    st.markdown("## Configuration")

    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        value = os.getenv("HF_API_KEY", ""),
        help="Enter your HuggingFace API token here. It stays local - never sent anywhere except HuggingFace.",
    )

    model_choice = st.selectbox(
        "Choose AI Model",
        options= list(AVAILABLE_MODELS.keys()),
        index=0,
        help = "All models are free via HuggingFace Interface API.",
    )
    st.divider()

    # How to get a free token 
    with st.expander(" Free Token kaise banao?"):
        st.markdown("""
1. jao [huggingface.co](https://huggingface.co) -> **Sign up** (free)
2. Top-right avatar -> **Settings**
3. Left menu -> **Access Tokens**
4. ** New Token** -> Type = **Read** -> create
5. copy karo -> upar paste karo                                                                                
                    """ )
        
    st.divider()

    # Tech Stack info 
    st.markdown('### Tech Stack')
    st.markdown("""
- **Langchain** - RAG pipeline
- **FAISS** - Vector similarity search 
- **sentence-transformer** - Local embeddings
- **HuggingFace API** - Free LLM Interface 
- **Streamlit** - User interface
    """)
    st.caption("100% open-source. No data stored")



#----------------------------------
# Main - File upload
# ----------------------------------
#
col_upload, col_info = st.columns([2,1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT.",
    )
with col_info:
    st.markdown("""
                ***Suported Formats***
                -PDF(most common)
                -DOCX(word document)
                -TXT(plain text)
                """)

# ------------------------
# Resume Text Extraction
#---------------------------

resume_text = None
if uploaded_file is not None:
    try:
        resume_text = extract_text_from_resume(uploaded_file)
        st.success(f"Resume loaded - {len(resume_text):,} characters extracted from **{uploaded_file.name}**")

        with st.expander(" Preview Extracted Text"):
            st.text_area(
                label="Extracted Resume content",
                value=resume_text[:3000],
                height=220,
                disabled=True,
            )

            if len(resume_text) > 3000:
                st.caption(f"Showing first 3,000 of {len(resume_text):,} characters.")

    except Exception as error:
    
            st.error(f"Could not read file: {error}")
            st.stop()