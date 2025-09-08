import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

import asyncio
import nest_asyncio

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def get_csv_text(file_path):
    """
    Extracts and aggregates every column from the CSV file into a structured content field.
    """
    all_texts = []
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return all_texts

    try:
        df = pd.read_csv(file_path)
        required_columns = [
            'case_title', 'date', 'court', 'jurisdiction',
            'author', 'bench', 'appellant_petitioner', 'respondent',
            'critical_info', 'acts_used', 'filename'
            ]


        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return all_texts

        for index, row in df.iterrows():
            content = f"""
Case Title: {row['case_title']}
Date: {row['date']}
Court: {row['court']}
Jurisdiction: {row['jurisdiction']}
Author: {row['author']}
Bench: {row['bench']}
Appellant / Petitioner: {row['appellant_petitioner']}
Respondent: {row['respondent']}
Critical Info: {row['critical_info']}
Filename: {row['filename']}
Acts Used: {row['acts_used']}
"""
            all_texts.append({"source": f"row_{index}", "content": content.strip()})
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

    return all_texts


def get_text_chunks(text):
    """Splits a large string of text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_or_create_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index_path = "faiss_index_2024"
    if os.path.exists(faiss_index_path):
        st.info("Loading existing vector store from disk.")
        return FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        st.info("Creating new vector store from CSV data... This may take a moment.")
        # Load all three CSV files
        documents = []
        for year_file in ["judgements_2024.csv","judgements_2023.csv","judgements_2022.csv","judgements_2021.csv","judgements_2020.csv","judgements_2019.csv"]:  # Only the new file
            docs = get_csv_text(year_file)
            if docs:
                documents.extend(docs)

        if not documents:
            st.warning("No text data found in the CSV files.")
            return None

        if not documents:
            st.warning("No text data found in the CSV file.")
            return None
        
        all_chunks = []
        for doc in documents:
            chunks = get_text_chunks(doc['content'])
    
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata={"source": doc["source"]}))

        vector_store = FAISS.from_documents(all_chunks, embedding=embeddings)
        vector_store.save_local(faiss_index_path)
        st.success("Vector store created successfully!")
        return vector_store

@st.cache_resource
def get_retrieval_qa_chain(llm, _retriever):
    prompt_template = """
You are a highly skilled and expert legal research assistant specialized in Supreme Court and High Court judgments.

Your task is to provide well-structured, clear, and accurate answers based solely on the provided context from judgments.

Instructions:
- Carefully analyze the context and synthesize a precise answer in your own words.
- Do NOT copy large sections verbatim from the context.
- If relevant information is missing in the context, clearly respond: "Not available."
- If the context does not cover the question at all, respond: "The context does not provide sufficient information."
- Use formal legal language and maintain a professional tone.
- Present the answer in a structured format (e.g., use numbered points or sections where appropriate).
- Provide reasoning and cite exact case details (case title, court, date, filename) if available.
- Summarize in 2 to 4 concise points.
- Restrict answers strictly to the retrieved context.


Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    case_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return case_qa

# --- Streamlit Application ---
st.set_page_config(page_title="Legal Assistant", layout="wide")
st.title("📜 Supreme Court Legal Research Assistant")

vector_store = get_or_create_vector_store()

if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
    case_qa = get_retrieval_qa_chain(llm, retriever)

    query = st.text_input("Enter your legal question:")

    if st.button("Get Answer") and query.strip():
        with st.spinner("Generating answer..."):
            try:
                result = case_qa.invoke({"query": query})
                
                st.subheader("✅ Answer:")
                st.write(result['result'])
                
                st.subheader("📚 Source Documents Preview:")
                for i, doc in enumerate(result['source_documents']):
                    st.write(f"--- Source Document {i + 1} ---")
                    st.write(doc.page_content[:500])  # Show first 500 characters
            except Exception as e:
                st.error(f"Error during Q&A: {e}")
else:
    st.error("Failed to initialize vector store.")