import streamlit as st
import tempfile
from analysir import resumeAnalyzer
from hrmode import hrResumeAnalyzer

# --- Sidebar Mode Selection ---
st.sidebar.title("Mode Selection")
mode = st.sidebar.radio("Choose Mode", ["HR Mode", "Candidate Mode"])

# --- App Title ---
st.title("AI Resume Analyzer ✦")
st.write("Upload a resume to get summary, pros/cons, and insights.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_info = ""  # For HR mode

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # --- HR Mode Job Description Input ---
    if mode == "HR Mode":
        job_info = st.text_area("Enter Job Description for HR Mode")

    # Determine if analyze button should be disabled
    analyze_disabled = (mode == "HR Mode" and not job_info.strip())

    if analyze_disabled:
        st.warning("Please enter the job description to analyze the resume.")

    # --- Analyze Button ---
    if st.button("Analyze Resume", disabled=analyze_disabled):
        with st.spinner("Analyzing resume..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Call the appropriate function based on mode
            if mode == "Candidate Mode":
                output = resumeAnalyzer(tmp_file_path)
            else:
                output = hrResumeAnalyzer(tmp_file_path, job_info)

            # --- Tabs for Results ---
            tabs = st.tabs(["Summary", "Pros / Cons", "Red Flags"])

            # Summary Tab
            with tabs[0]:
                st.subheader("Resume Summary")
                st.write(output.get("summary", "No summary available."))

            # Pros / Cons Tab
            with tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Pros ✅")
                    st.write(output.get("pros", "No pros available."))
                with col2:
                    st.subheader("Cons ❌")
                    st.write(output.get("cons", "No cons available."))

            # Red Flags Tab
            with tabs[2]:
                st.subheader("Red Flags ⚠️")
                st.write(output.get("red_flags", "No red flags detected."))
st.markdown(
    """
    <div style='
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: gray;
        font-size: 12px;
    '>
        © 2025 Rudra Yadav - AI Resume Analyzer
    </div>
    """,
    unsafe_allow_html=True
)
    

