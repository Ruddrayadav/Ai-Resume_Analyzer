import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


def hrResumeAnalyzer(file_path , job_info):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
   

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len
    )

    docs = text_splitter.split_documents(documents)
    resume_text = " ".join([doc.page_content for doc in docs])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
      
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./resume_db"
    )

    retriever = vector_store.as_retriever()

    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key = os.getenv("GOOGLE_API_KEY")
    )
    hr_analysis_template = """
    You are an expert Talent Acquisition Specialist and Technical Recruiter with a keen eye for detail. Your primary task is to conduct a thorough, unbiased analysis of a candidate's resume against a specific job requirement. Your goal is to provide a hiring manager with a clear, actionable summary that saves them time and highlights the most critical information for their decision-making process.

    **Inputs:**
    1.  **Job Requirements:** {job_description}
    2.  **Candidate's Resume:** {resume_text}

    **Instructions:**
    Based on the provided inputs, generate a concise "Candidate Fit Report" using the exact Markdown structure below. Do not add any conversational text outside of this structure.

    ---

    ### Candidate Fit Report: [Candidate's Likely Name] for [Job Title from Requirements]

    **1. Executive Summary:**
    Provide a 3-4 sentence top-line summary. Start with a direct verdict on the candidate's suitability (e.g., "Strong Match," "Partial Match," "Potential Fit," "Not a good fit"). Mention the candidate's years of relevant experience and highlight the 1-2 most significant areas of alignment or misalignment with the job requirements.

    **2. Requirement-by-Requirement Breakdown:**
    Create a table that lists the top 5-7 most critical requirements from the job description. For each requirement, assess the candidate's resume, cite the specific evidence (or lack thereof), and assign a "Match Level" (High, Medium, Low, None).

    | Requirement | Evidence from Resume | Match Level |
    | :--- | :--- | :--- |
    | **[Skill/Experience 1]** | [Quote the relevant line or summarize the experience] | [High/Medium/Low/None] |
    | **[Skill/Experience 2]** | [Quote the relevant line or summarize the experience] | [High/Medium/Low/None] |
    | **[Years of Experience]** | [State the years mentioned in the resume] | [High/Medium/Low/None] |
    | ... | ... | ... |

    **3. Strengths (Direct Matches):**
    In a bulleted list, highlight the 3-4 strongest qualifications or experiences from the resume that directly align with the most important duties in the job description.
    * **Strength 1:** [Example: Possesses 5+ years of Python experience, directly matching the core requirement.]
    * **Strength 2:** [Example: Led a project that resulted in a 15% cost reduction, demonstrating the required 'optimization' skill.]

    **4. Potential Gaps & Red Flags:**
    In a bulleted list, identify any significant skills, qualifications, or experiences mentioned in the job description that are **missing** from the resume. Also, note any potential red flags (e.g., unexplained employment gaps, frequent job hopping, potential skill exaggeration, typos/errors).
    * **Gap 1:** [Example: The role requires experience with 'Azure DevOps', which is not mentioned in the resume.]
    * **Red Flag 1:** [Example: Candidate has changed jobs every 10-12 months for the past 4 years, which may indicate a retention risk.]

        **5. Final Recommendation:**
    Conclude with a clear recommendation.
     **Verdict:** (e.g., "Recommend for Interview," "Proceed with Caution," "Reject")
    - **Reasoning:** (Briefly justify your verdict based on the analysis above.)

    ---
    """
    hr_chain = LLMChain(llm = llm , prompt = PromptTemplate(input_variables=["job_description","resume_text"],template=hr_analysis_template))
 
    qa = RetrievalQA.from_chain_type(llm = llm , retriever=retriever )
    
    answer = hr_chain.run({
             "job_description": job_info,
             "resume_text": resume_text
    })
    return {
        "summary": answer,
        "pros": "",
        "cons": "",
        "red_flags": ""
    }







