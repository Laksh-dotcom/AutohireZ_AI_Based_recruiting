# AutohireZ_AI_Based_recruiting
---

An AI-powered, local-friendly resume checker that uses lightweight LLM agents to match candidates with job descriptions intelligently and efficiently and if score is greater than or equal to the threshold score mail for next process is sent automatically.

---

## Features

- **PDF Resume Parsing** – Extract structured data from raw resumes.
- **Multi-Agent AI System** – Modular agents using LangChain to handle summarization, matching, and reasoning.
- **Local LLMs via Ollama** – Runs on small models (Gemma 1B, DeepSeek 1.5B) to support low-resource systems.
- **Semantic Matching** – Embedding-based cosine similarity between job descriptions and resumes.
- **SQLite Integration** – Stores candidate and job info with ease.
- **SMTP Intergration** - Sends the mail automatically.
---

## Tech Stack

| Component        | Tool/Library                |
|------------------|-----------------------------|
| LLMs             | Ollama (Gemma3: 1b)         |
| Agent Framework  | LangChain                   |
| Resume Parsing   | PDFMiner                    |
| DB               | SQLite                      |
| Embeddings       | Ollama                      |
| Language         | Python                      |

---

## System Architecture

1. **Resume Parser** → Extracts raw data from PDFs.
2. **Summarization Agent** → Generates a summary of candidate profiles.
3. **Job Description Agent** → Processes job descriptions similarly.
4. **Similarity Agent** → Uses embeddings to compare resumes vs jobs.
5. **Result Generator** → Outputs the best-fit candidates with match percentage.
6. **Send Email automatically** → Sends file automatically.
---
