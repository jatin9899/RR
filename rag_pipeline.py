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

<<<<<<< HEAD
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chains import RetrievalQA
from langchain_prompts import PromptTemplate
=======
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# helper to validate model access
try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None

AVAILABLE_MODELS = {
    "Mistral 7B Instruct (Recommended)": "mistralai/mistral-7b-instruct",
}

# Public fallback model to use when HF access fails or token is missing
FALLBACK_MODEL = "google/flan-t5-large"
>>>>>>> 4456eaa (final_second_commit)

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
<<<<<<< HEAD
        self._qa_chain = None
=======
        # If no token provided, force the public fallback model
        if not self.hf_token:
            self.model_id = FALLBACK_MODEL
        self._qa_chain = None
        self._resume_text: Optional[str] = None
        self._use_heuristic = False
>>>>>>> 4456eaa (final_second_commit)


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
<<<<<<< HEAD
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            )
=======
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError as imp_err:
            raise RuntimeError(
                "Missing dependency: sentence-transformers (or its transitive deps). "
                "Install with: pip install sentence-transformers faiss-cpu\n"
                "If you're on Windows and encounter issues installing faiss, see: https://github.com/facebookresearch/faiss/wiki/Installing-Faiss"
            ) from imp_err
>>>>>>> 4456eaa (final_second_commit)
        
        # Step 3 : Create FAISS vector store (in-memory)
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )
        retriever = vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs={"k": 4}
        )

<<<<<<< HEAD
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
=======
        # Step 4 : Initialize the HuggingFace LLM via Interface API
        # If no HF token is provided, fall back to a simple local heuristic
        # responder to avoid calling the remote inference API and hitting
        # provider-mapping StopIteration errors.
        if self.hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_token
        else:
            os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
            self._use_heuristic = True

        # Validate model access early. If the requested model is gated, the
        # token invalid, or there are no inference providers for the model,
        # try a public fallback. This avoids a StopIteration when the
        # inference provider mapping is empty.
        fallback_model = FALLBACK_MODEL
        if HfApi is not None and self.hf_token:
            api = HfApi()
            try:
                info = api.model_info(self.model_id, expand=["inferenceProviderMapping"])
                # Inspect raw dict for the provider mapping robustness across hf versions
                try:
                    info_dict = info.to_dict()
                except Exception:
                    info_dict = getattr(info, "_asdict", lambda: {})() or {}

                provider_mapping = (
                    info_dict.get("inferenceProviderMapping")
                    or info_dict.get("inference_provider_mapping")
                    or info_dict.get("inference", {}).get("inferenceProviderMapping")
                    or {}
                )

                if not provider_mapping:
                    # model cannot be served via HF Inference API for this account
                    # attempt fallback
                    api.model_info(fallback_model)
                    self.model_id = fallback_model

            except Exception as info_err:
                # Couldn't access requested model metadata; try fallback and
                # surface a clear error if fallback also fails.
                try:
                    api.model_info(fallback_model)
                    self.model_id = fallback_model
                except Exception:
                    raise RuntimeError(
                        f"Unable to access Hugging Face model '{self.model_id}'. "
                        f"Original error: {info_err}.\nEnsure your Hugging Face token is correct, "
                        "has 'read' permission, and that the model is accessible to your account."
                    )

        if not self._use_heuristic:
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
        else:
            # heuristic mode: no remote LLM will be used
            self._qa_chain = None
        # keep the resume text so we can rebuild using a fallback model if needed
        self._resume_text = resume_text
>>>>>>> 4456eaa (final_second_commit)




    #------------QUERIES---------------------

<<<<<<< HEAD
    def ask(self, question: str) -> str:
        """
        Internal helper to run a query through the RAG chain
        """
        if self._qa_chain is None:
            raise RuntimeError("Pipeline not built yet. Call build(resume_text) First.")
        result = self._qa_chain.run(query=question)
        return result.get("result", "").strip()
    
    def get_rating(self) -> str:
=======
    def ask(self, question: str, role: Optional[str] = None) -> str:
        """
        Internal helper to run a query through the RAG chain.
        """
        if self._qa_chain is None:
            if self._use_heuristic and self._resume_text:
                return self._heuristic_answer(question, self._resume_text)
            raise RuntimeError("Pipeline not built yet. Call build(resume_text) first.")
        try:
            result = self._qa_chain.run(query=question)
        except Exception as err:
            # Detect StopIteration and other HF provider errors robustly.
            fallback_model = FALLBACK_MODEL
            def _is_provider_error(e: Exception) -> bool:
                if isinstance(e, StopIteration):
                    return True
                name = e.__class__.__name__
                if name in ("StopIteration", "RepositoryNotFoundError"):
                    return True
                text = str(e) or ""
                if "inferenceProviderMapping" in text or "No inference providers" in text:
                    return True
                # check chained exceptions
                cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
                if isinstance(cause, StopIteration):
                    return True
                return False

            if _is_provider_error(err):
                # switch to heuristic responder if remote call fails or no token
                if self._resume_text:
                    self._use_heuristic = True
                    return self._heuristic_answer(question, self._resume_text, role=role)
                else:
                    raise RuntimeError(
                        "LLM provider error and no resume text available to rebuild. "
                        "Provide a valid Hugging Face token or try a public model in AVAILABLE_MODELS."
                    )
            else:
                raise

        if isinstance(result, dict):
            result_text = result.get("result", "")
        else:
            result_text = result
        return result_text.strip()
    
    def get_rating(self, role: Optional[str] = None) -> str:
>>>>>>> 4456eaa (final_second_commit)
        """
        Get an overall rating for the resume 
        """
        return self.ask(
            "Rate this resume on a scale of 1 to 10."
            "Start with 'Score: X/10' on the first line,"
            "then give a 2-3 sentence justification for the score."
<<<<<<< HEAD
        )
    
    def get_strengths(self) -> str:
=======
        , role=role)

    def _heuristic_answer(self, question: str, resume_text: str, role: Optional[str] = None) -> str:
        """Simple heuristic responder used when no HF token / provider available.
        This is intentionally lightweight and deterministic.
        """
        q = question.lower()
        txt = resume_text.lower()
        words = txt.split()
        length = len(words)

        # rating heuristic: prefer longer resumes and presence of keywords
        keywords = ["python", "aws", "sql", "machine learning", "ml", "lead", "manager", "data"]
        kw_count = sum(1 for k in keywords if k in txt)
        score = min(10, max(3, int((length / 200.0) + kw_count)))
        if "rate" in q or "score" in q:
            role_note = f" for role: {role}" if role else ""
            return f"Score: {score}/10\nThis is a heuristic score based on resume length and keyword matches{role_note}. Provide a HuggingFace token for richer analysis."

        if "strength" in q:
            strengths = []
            for k in keywords:
                if k in txt:
                    strengths.append(f"Has experience with {k}")
            if not strengths:
                strengths = ["Concise summary", "Relevant formatting"]
            return "\n".join([f"- {s}" for s in strengths[:5]])

        if "what is missing" in q or "weak" in q or "missing" in q:
            weaknesses = []
            if "education" not in txt:
                weaknesses.append("Missing or unclear Education section")
            if "experience" not in txt and "work" not in txt:
                weaknesses.append("Missing detailed Work / Experience history")
            if not weaknesses:
                weaknesses = ["Be more specific about achievements (add metrics)"]
            return "\n".join([f"- {w}" for w in weaknesses[:6]])

        if "suggest" in q or "improv" in q:
            suggestions = [
                "Add specific metrics for accomplishments (e.g., reduced X by Y%)",
                "Include a clear Education section with dates",
                "List technologies used for each role (Python, SQL, AWS, etc.)",
            ]
            return "\n".join([f"{i+1}. {s}" for i, s in enumerate(suggestions)])

        # default fallback
        return "I couldn't run the remote model. Provide a HuggingFace API token for full AI analysis, or use these quick heuristics."
    
    def get_strengths(self, role: Optional[str] = None) -> str:
>>>>>>> 4456eaa (final_second_commit)
        """
        Get the top 3 strengths of the resume
        """
        return self.ask(
            "List the top 5 strengths and positive highlights of this resume."
            "Use bullet points. Be specific - mention actual skills,"
            "achievements, and experiences found in the resume."
<<<<<<< HEAD
        )
    
    def get_weaknesses(self) -> str:
=======
        , role=role)
    
    def get_weaknesses(self, role: Optional[str] = None) -> str:
>>>>>>> 4456eaa (final_second_commit)
        """
        Get the top 3 weaknesses of the resume
        """
        return self.ask(
            "What is missing or weak in this resume?"
            "List all the gaps, missing sections, vage descriptions,"
            "or areas that need improvement as bullet points."
            "Be honest and critical."
<<<<<<< HEAD
        )
    
    def get_suggestions(self) -> str:
=======
        , role=role)
    
    def get_suggestions(self, role: Optional[str] = None) -> str:
>>>>>>> 4456eaa (final_second_commit)
        """
        Get specific suggestions to improve the resume
        """
        return self.ask(
            "Give 6 specific, actionable suggestions to improve this resume."
            "Number each suggestion 1 tp 6."
            "Be concrete - explain WAHT to add, HOW to rewrite, or WHAT to remove."
<<<<<<< HEAD
        )
    
    def get_ats_score_feedback(self) -> str:
=======
        , role=role)
    
    def get_ats_score_feedback(self, role: Optional[str] = None) -> str:
>>>>>>> 4456eaa (final_second_commit)
        """
        Get feedback on the resume's ATS (Applicant Tracking System) score and how to improve it
        """
        return self.ask(
            "Evaluate this resume's ATS (Applicant Tracking System) score out of 100."
            "Cover: keywords densitym formatting issues, section headings,"
            "and give 3 specific keywords the candidates should add."
            "Start with 'ATS Score: X/100'."
<<<<<<< HEAD
        )
=======
        , role=role)
>>>>>>> 4456eaa (final_second_commit)
