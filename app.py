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

    # Read server-side token (if set) but do NOT pre-fill it into the browser field.
    # This prevents the token from appearing in page source or being exposed to users.
    env_hf_token = os.getenv("HF_API_KEY", "")
    hf_token_input = st.text_input(
        "HuggingFace API Token",
        type="password",
        value = "",
        help="Paste your HuggingFace API token here (kept hidden). Leave blank to use a server-side token if configured.",
    )
    # Use the user input token if provided; otherwise fall back to the server-side token.
    hf_token = hf_token_input.strip() or env_hf_token

    model_choice = st.selectbox(
        "Choose AI Model",
        options= list(AVAILABLE_MODELS.keys()),
        index=0,
        help = "All models are free via HuggingFace Interface API.",
    )
    st.divider()

    role_choice = st.selectbox(
        "Target role / Job area",
        options=[
            "Frontend Engineer",
            "Backend Engineer",
            "AI Engineer",
            "Data Scientist",
            "Product Manager",
            "Other (custom)",
        ],
        index=2,
        help="Select the role to tailor the resume feedback.",
    )
    if role_choice == "Other (custom)":
        role = st.text_input("Enter custom role", value="")
    else:
        role = role_choice
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

#------------------------
# Analyze Button
# ------------------------

if resume_text:
    st.divider()
    analyze_btn = st.button(
        "Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled= not hf_token,
    )

    if not hf_token:
        st.warning("Enter your HuggingFace API token in the sidebar to enable analysis.")

    if analyze_btn and hf_token:
        # -------BUILD RAG PIPELINE--------
        pipeline = ResumeRAGPipeline(hf_token=hf_token, model_name=model_choice)

        with st.spinner("Buulding RAG pipeline - chunking, embedding,  indexing... (one-time setup)"):
            try:
                pipeline.build(resume_text)
                st.success("RAG pipeline ready!")
            except Exception as error:
                st.error(f"pipeline error:{error}")    
                st.stop()

        st.markdown("## Resume Analysis Report")
        st.markdown(f"*Model selected: **{pipeline.model_id}** (choice: {model_choice}) | File: **{uploaded_file.name}***")
        st.divider()

        #------Row 1 : Rating + ATS ---------------
        # __ Row 1: Rating + ATS __
        col_rating, col_ats = st.columns(2)

        with col_rating:
            st.markdown("### ⭐ Overall Rating")
            with st.spinner("Rating your resume..."):
                rating_text = pipeline.get_rating(role=role)
            st.markdown(
                f"<div class='card'>{rating_text}</div>",
                unsafe_allow_html=True,
            )

        with col_ats:
            st.markdown("### 🖥️ ATS Compatibility")
            with st.spinner("Checking ATS compatibility..."):
                ats_text = pipeline.get_ats_score_feedback(role=role)
            st.markdown(
                f"<div class='card card-purple'>{ats_text}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # __ Row 2: Strengths + Weaknesses __
        col_strong, col_weak = st.columns(2)

        with col_strong:
            st.markdown("### 💪 Strengths")
            with st.spinner("Finding strengths..."):
                strengths_text = pipeline.get_strengths(role=role)
            st.markdown(
                f"<div class='card card-green'>{strengths_text}</div>",
                unsafe_allow_html=True,
    )

        with col_weak:
            st.markdown("### ⚠️ What's Missing / Weak Points")
            with st.spinner("Identifying gaps..."):
                weaknesses_text = pipeline.get_weaknesses(role=role)
            st.markdown(
                f"<div class='card card-red'>{weaknesses_text}</div>",
                unsafe_allow_html=True,
    )

        st.divider()

        # __ Row 3: Suggestions (full width) __
        st.markdown("### 💡 Improvement Suggestions")
        with st.spinner("Generating personalised suggestions..."):
            suggestions_text = pipeline.get_suggestions(role=role)
        st.markdown(
            f"<div class='card card-orange'>{suggestions_text}</div>",
            unsafe_allow_html=True,
)

        st.divider()

        # __ Download full report __
        full_report = f"""### Resume Analysis Report
    Model: {model_choice}
    File: {uploaded_file.name}

    ## Overall Rating
    {rating_text}

    ## ATS Compatibility
    {ats_text}

    ## Strengths
    {strengths_text}

    ## What's Missing / Weak Points
    {weaknesses_text}

    ## Improvement Suggestions
    {suggestions_text}
    """

        st.download_button(
            label="📄 Download Full Report (.md)",
            data=full_report,
            file_name="resume_analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

elif uploaded_file is None:
    st.info("📤 Upload your resume above to get started.")        