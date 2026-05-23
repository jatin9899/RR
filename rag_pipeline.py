"""
rag_pipeline.py
----------------
Langchain RAG pipeline for processing resume text and generating feedback.

flow:
   Resume Text
      |
   RescursiveCharacterTextSplitter (chunking)
      |
   HuggingFaceEmbeddings (embedding)    (local sectence-transformers, no API needed)
      |
   FAISS Vector Store                    (in-momory similarity search)
      |
   Retriever  ---> HuggingFace Interface API LLM
      |
   Structured Resume Analysis
"""

import os
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chains import RetrievalQA
from langchain_prompts import PromptTemplate

# --------------------------------------
# Prompt Template
# --------------------------------------

_BASE_TEMPLATE = """you are an expert HR recruiter and resume reviewer with 15+ years of experience hiring across software engineering, data science, and product management roles.

Use Only the resume content below to answer the question.
Be honest, specific, and actionable

------RESUME CONTENT------
{context}
--------------------------

Question: {question}
Answer: """

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_BASE_TEMPLATE,
)

# --------------------------------------
# Core Pipeline Class
# --------------------------------------

class ResumeRAGPipeline:
    """
    End-to-end Langchain RAG pipeline for resume review.
    Usage :
    pipeline = ResumeRAGPipeline(hf_token="hf_...", model_name="Mistral 7B Instruct")
    pipeline.build(resume_text)
    
    rating       = pipeline.get_rating()
    strengths    = pipeline.get_strengths()
    weaknesses   = pipeline.get_weaknesses()
    suggestions    = pipeline.get_suggestions()
    ats_score    = pipeline.get_ats_score_feedback()
    """
    def __init__(
        self,
        hf_token: str,
        model_name: str = "Mistral 7B Instruct (Recommended)"):
        self.hf_token = hf_token.strip()
        self.model_id = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS["Mistral 7B Instruct (Recommended)"])
        self._qa_chain = None


    #------------BUILD--------------------------------
    def build(self, resume_text: str) -> None:
        """
        Ingest resume text -> chunk -> embed -> vector store -> retriever -> RetrievalQA chain
        call this once before querying
        """

        # Step 1 : Chunk the resume
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80, separators=["\n\n", "\n", " ", ""],)
        documents = splitter.create_documents([resume_text])

        # step2 : Embef using local sentence-transformers model (no API calls token needed for this step)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            )
        
        # Step 3 : Create FAISS vector store (in-memory)
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )
        retriever = vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs={"k": 4}
        )

        # Step 4 : Initialize the HuggingFace LLM viva Interface API
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_token

        llm = HuggingFaceEndpoint(
            repo_id=self.model_id,
            huggingfacehub_api_token=self.hf_token,
            temperature=0.3,
            max_new_tokens=600,
            timeout=180,
        )

        # Step 5 : Create the RetrievalQA chain
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = "stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": _PROMPT},
        )




    #------------QUERIES---------------------

    def ask(self, question: str) -> str:
        """
        Internal helper to run a query through the RAG chain
        """
        if self._qa_chain is None:
            raise RuntimeError("Pipeline not built yet. Call build(resume_text) First.")
        result = self._qa_chain.run(query=question)
        return result.get("result", "").strip()
    
    def get_rating(self) -> str:
        """
        Get an overall rating for the resume 
        """
        return self.ask(
            "Rate this resume on a scale of 1 to 10."
            "Start with 'Score: X/10' on the first line,"
            "then give a 2-3 sentence justification for the score."
        )
    
    def get_strengths(self) -> str:
        """
        Get the top 3 strengths of the resume
        """
        return self.ask(
            "List the top 5 strengths and positive highlights of this resume."
            "Use bullet points. Be specific - mention actual skills,"
            "achievements, and experiences found in the resume."
        )
    
    def get_weaknesses(self) -> str:
        """
        Get the top 3 weaknesses of the resume
        """
        return self.ask(
            "What is missing or weak in this resume?"
            "List all the gaps, missing sections, vage descriptions,"
            "or areas that need improvement as bullet points."
            "Be honest and critical."
        )
    
    def get_suggestions(self) -> str:
        """
        Get specific suggestions to improve the resume
        """
        return self.ask(
            "Give 6 specific, actionable suggestions to improve this resume."
            "Number each suggestion 1 tp 6."
            "Be concrete - explain WAHT to add, HOW to rewrite, or WHAT to remove."
        )
    
    def get_ats_score_feedback(self) -> str:
        """
        Get feedback on the resume's ATS (Applicant Tracking System) score and how to improve it
        """
        return self.ask(
            "Evaluate this resume's ATS (Applicant Tracking System) score out of 100."
            "Cover: keywords densitym formatting issues, section headings,"
            "and give 3 specific keywords the candidates should add."
            "Start with 'ATS Score: X/100'."
        )