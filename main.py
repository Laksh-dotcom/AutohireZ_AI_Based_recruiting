import pandas as pd
import ollama
import numpy as np
import sqlite3
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from pdfminer.high_level import extract_text
from langchain_community.utilities import SQLDatabase
from sklearn.metrics.pairwise import cosine_similarity
from send import Details

"""*LOAD DATA"""
jd_summary = ""
threshold=50
resume_summary = ""
company_name=""
resume_path=r"C:/Users/karan/OneDrive/Desktop/sandeep.pdf"
resume_text = extract_text(resume_path)
job_title="Sales Coordinator"
job_descriptions="""
Job Title: Sales Coordinator
Company: VoltEdge Solutions Pvt. Ltd.
Location: Noida, Uttar Pradesh
Salary: ₹25,000 – ₹35,000 per month
Department: Sales and Business Development
Reports To: Regional Sales Manager

About the Company:
VoltEdge Solutions Pvt. Ltd. is a growing player in the electrical and automation solutions industry, catering to both retail and commercial clients across North India. With a focus on innovation, service excellence, and customer-centricity, we aim to empower businesses with reliable electrical solutions.

Job Overview:
We are seeking a proactive and detail-oriented Sales Coordinator to support our regional sales team. This role involves managing client communication, handling sales operations, and ensuring seamless coordination between departments to meet sales targets and customer satisfaction benchmarks.

Key Responsibilities:

Assist the sales team in preparing quotations, sales orders, and invoices.
Manage client communications—handle inquiries, provide support, and maintain long-term relationships.
Coordinate with internal departments (logistics, finance, procurement) to ensure smooth execution of orders.
Maintain and update sales records, client databases, and CRM systems.
Prepare regular reports and dashboards using MS Excel (including PivotTables, charts, and formulas).
Schedule meetings, maintain calendars, and provide administrative support to the sales team.
Help organize sales campaigns, exhibitions, and client visits.


Required Skills & Qualifications:

BBA (Marketing/HR) – final semester or recent graduate.
Proficient in MS Office, especially Excel and Word.
Strong written and verbal communication skills.
Good coordination, time management, and follow-up skills.
Ability to multitask and stay organized under pressure.
Prior internship or sales admin experience (preferred but not mandatory)."""

"""CEATING AND CONNECTING SQL DATABASE USING LANGCHAIN*"""

def connect_to_database(name, e_mail, phone_no, score):
    con = sqlite3.connect('CANDIDATE_RESUME.db')
    db = SQLDatabase.from_uri("sqlite:///CANDIDATE_RESUME.db")
    cursor = con.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            Name TEXT NOT NULL,
            Email TEXT NOT NULL PRIMARY KEY,
            Phone_no TEXT NOT NULL,
            Score INTEGER NOT NULL
        )
    ''')
    cursor.execute('INSERT INTO users (Name, Email, Phone_no, Score) VALUES (?, ?,?,?)', (name, e_mail, phone_no, score))
    con.commit()


"""INITIALIZE LLM*"""

llm_gemma = ChatOllama(model="gemma3:1b")


def summarize_jd(job_desc: str) -> str:
    
    prompt = f"""Summarize this job description in 3-5 bullet points only key skills and Qualifications:
    - Key skills:  
    - Qualifications:  
    - Responsibilities:  
    ---  
    {job_desc}"""
    responce = ''
    for chunk in llm_gemma.stream(prompt):
        responce += chunk.content
    return responce

def summarize_resume(resume:str) ->str:

    prompt = f"""You need to summarize the given resume and tell me the key points out of it which are:
    -Education:
    -Exprience:
    -Skills:
    -Certifiaction:
    ---
    {resume}"""
    resume_summarized = ''
    for chunk in llm_gemma.stream(prompt):
        resume_summarized += chunk
    return resume_summarized

def get_embedding(text):
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)  # Use Gemma/Mistral for better context
    return np.array(response['embedding'])

def compare(jd_summary, resume_summary):
    jd_emmbd = get_embedding(jd_summary)
    resume_embd = get_embedding(resume_summary)
    similarity = cosine_similarity([jd_emmbd], [resume_embd])[0][0]*100
    return similarity

def candidate_results(resume):
    prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    Extract the following details from the resume below:

    Resume Text:
    {resume}

    Extracted Info:
    - Name:
    - Email:
    - Phone Number:

    Output in JSON format.
    """
    )
    llm = Ollama(model="gemma3:1b")
    chain = LLMChain(prompt=prompt, llm=llm)
    response = chain.run(resume=resume_text) 
    data=response.split(",")
    main_data=[]
    for i in range(len(data)):
        content=data[i].split(":")
        for i,val in enumerate(content):
            if i==1:
                main_data.append(val)
                if len(main_data)==3:
                    break
    total=[]
    for i,val in enumerate(main_data):
        if i==2:
            new_val=val.split("\n}\n")
            total.append(new_val[0][2:len(new_val[0])-1])
            continue
        total.append(val[2:len(val)-1])
    return total
"""*BELOW IS SUMMARIZER TOOL THAT SUMMARIZES THE JOB DESCRIPTION..."""

summarizeJD_tool = Tool(
    name="JobDescriptionSummarizer",
    func=summarize_jd,
    description="Summarizes job descriptions into key skills and qualifications."
)
"""*USING THE AGENT BELOW WE ARE GOING TO SUMMARIZE CANDIDATES'S RESUME."""
summarize_resume_tool = Tool(
    name = "Resume Summarizer of the candidate",
    func = summarize_resume,
    description="Here you summarize the resume of candidate"
)

"""*THIS IS NOTHING BUT AGENT CHAIN THAT WE ARE GONNA USE.."""
summarize_agent = initialize_agent(
    tools=[summarizeJD_tool, summarize_resume_tool],  # Register the tool
    llm=llm_gemma,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Use React-based decision-making
    verbose=True,
    handle_parsing_errors=True
)

jd_summary = summarize_agent.run(f"Give me keywords from job description in points: {job_descriptions}")

resume_summary = summarize_agent.run(f"Give me keywords from resume text in key points: {resume_text}") 

scores = int(compare(jd_summary, resume_summary))

details = candidate_results(resume_summary)

list_of_details = candidate_results(resume_summary)

name = list_of_details[0]
e_mail = list_of_details[1]
phone = list_of_details[2]

def create_message(name,job_title,company_name):
    return f"Dear {name}, we're pleased to invite you for an HR interview regarding the {job_title} position at {company_name}. Please confirm your availability."

print(scores)

if scores>=threshold:
    connect_to_database(name, e_mail, phone, scores)
    data=create_message(name,job_title,company_name)
    user_send=Details(name,e_mail,data)
    user_send.send()