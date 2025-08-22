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


def resumeAnalyzer(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len
    )

    docs = text_splitter.split_documents(documents)

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

    summary_template = """
    You are a highly experienced career analyst and professional resume writer, specializing in creating high-impact summaries for busy hiring managers. Your goal is to distill the most critical information from a resume into a concise and easy-to-scan format.

    From the resume text provided below, generate a professional summary structured exactly as follows:

    **1. Narrative Summary:**
    A brief paragraph (3-4 sentences) that encapsulates the candidate's professional identity, core expertise, and years of relevant experience. Start with their most likely professional title (e.g., "A results-oriented Senior Software Engineer...").

    **2. Key Qualifications:**
    A bulleted list of the 3-5 most compelling skills, technologies, or qualifications that make this candidate stand out. Focus on tangible and in-demand skills mentioned in the resume.
    - Example Item 1
    - Example Item 2
    - ...

    **3. Quantifiable Achievements:**
    A bulleted list of 1-3 of the most impressive, data-backed achievements from the resume. If no quantifiable data exists, state "No specific quantifiable achievements were found."
    - Example: Increased team efficiency by 25% through the implementation of a new CI/CD pipeline.
     ...

    **4. Potential Target Roles:**
    Based on the skills and experience, list 2-3 specific job titles this candidate is well-suited for.

    **Resume Text:**
    {text}
    """



    pros_cons_template = """
    You are a seasoned hiring manager and career coach with over 15 years of experience. Your task is to provide a constructive, detailed, and actionable critique of the provided resume. The goal is to help the candidate significantly improve their resume to attract top employers in their field.

    Analyze the resume based on the following criteria: Clarity, Impact, ATS (Applicant Tracking System) Compatibility, and Overall Presentation.

    Provide your analysis in the following Markdown format:

    ### Strengths (Pros)
    Create a bulleted list of the resume's strongest points. For each point, briefly explain *why* it is a strength.
    * **Point 1:** [Example: Excellent use of quantifiable metrics.]
      * **Reason:** [Example: The candidate effectively uses numbers and percentages (e.g., 'increased sales by 20%') to demonstrate tangible impact, which is highly compelling to recruiters.]
    * **Point 2:** [Example: Clear and Professional Summary.]
      * **Reason:** [Example: The opening summary immediately communicates the candidate's experience level and key skills.]

    ### Areas for Improvement (Cons & Actionable Advice)
    Create a bulleted list of the resume's weaknesses. For each weakness, provide a **specific, actionable recommendation** for how to fix it. This is the most important section.
    * **Weakness 1:** [Example: Vague or passive language in job descriptions.]
      * **Recommendation:** [Example: Rephrase bullet points to start with strong action verbs (e.g., 'Led', 'Architected', 'Implemented', 'Negotiated'). Change 'Was responsible for managing a team' to 'Managed a team of 5 engineers to deliver the project 15% ahead of schedule.']
    * **Weakness 2:** [Example: Lacks keywords relevant to the target role.]
      * **Recommendation:** [Example: Research job descriptions for the target role (e.g., 'Data Scientist') and incorporate key terms like 'Python', 'TensorFlow', 'Scikit-learn', and 'ETL pipelines' naturally into the Skills and Experience sections to improve ATS scores.]

    ### Overall Verdict & Key Takeaway
    Conclude with a brief, one-paragraph summary of your overall assessment. State the single most critical change the candidate should make to elevate their resume.

    **Resume Text:**
    {text}
    """
    
    summary_chain = LLMChain(llm = llm , prompt = PromptTemplate(input_variables=["text"],template=summary_template))
    pros_chain = LLMChain(llm = llm , prompt = PromptTemplate(input_variables=["text"],template=pros_cons_template))

    qa = RetrievalQA.from_chain_type(llm = llm ,retriever=retriever)

    resume_text = " ".join([doc.page_content for doc in docs])

    summary_result = summary_chain.run(text=resume_text)
    pros_result = pros_chain.run(text=resume_text)

    return {
        "summary": summary_result,
        "pros": pros_result,
        "cons": pros_result,  # You could split pros_result if you want separate cons
        "red_flags": "No red flags detected."  # placeholder
    }

   







