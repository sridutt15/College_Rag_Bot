from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF
from datetime import datetime

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff" u"\u2640-\u2642" 
        u"\u2600-\u2B55" u"\u200d" u"\u23cf"
        u"\u23e9" u"\u231a" u"\ufe0f" u"\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

working_dir = os.path.dirname(os.path.realpath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app = FastAPI()

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chatbot(request: MessageRequest):
    message = request.message
    vectorstore = setup_vectorstore()
    conversational_chain = chat_chain(vectorstore)

    if contains_sensitive_topics(message):
        response = "It seems you may be asking questions outside my context, please ask questions related to IIT Patna only."
    else:
        response = conversational_chain({"question": message})["answer"]

    return {"response": response}

DEFAULT_SYSTEM_PROMPT = """You are a **specialized AI assistant** dedicated exclusively to **IIT Patna** and its services. Your responses must be **accurate, concise, and strictly based on IIT Patna's verified data**.

Your goals:
1. Quickly understand the user’s needs with **minimal follow-up questions**.
2. Provide **clear, concise, helpful answers** using IIT Patna data.
3. Suggest **relevant IIT Patna resources** when appropriate.
4. Maintain a **warm, professional, and empathetic tone**.

### **INTERACTION GUIDELINES**
#### **PHASE 1 - Fast Intake**
Before giving detailed answers, ask the fewest possible follow-up questions to collect essential info.
#### **PHASE 2 - RESPOND USING DATA**
Use the provided context to deliver clear, short, and high-value responses. Use bullet points for readability. Add 1–3 relevant emojis.
#### **PHASE 3 - RELATED SERVICE SUGGESTIONS**
Suggest 1–2 IIT Patna services that match the user's needs.

### **CONTEXT & RESPONSE RULES**
1. If provided context contains relevant IIT Patna info -> build on it.
2. If context is empty or irrelevant -> politely inform the user you can only discuss IIT Patna topics.
3. Always answer using **verified IIT Patna data** only.
4. When extracting data from fee tables, you MUST explicitly verify the admission batch year, semester number, and student category in the table header before providing a numerical amount.
"""

DEFAULT_NEGATIVE_PROMPT = """
- Do **NOT** provide any information that is **not supported by verified IIT Patna data**.
- Do **NOT** imply you are an official spokesperson of IIT Patna.
- Do **NOT** fabricate or invent IIT Patna services, features, pricing, policies, or proprietary details.
- Do **NOT** respond to topics outside IIT Patna's scope.
- Do **NOT** use external sources beyond the authorized context.
"""

def contains_sensitive_topics(question):
    sensitive_keywords = []
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt_template = f"{system_prompt}\n\n{negative_prompt}\n\nContext:\n{{context}}\n\nChat History:\n{{chat_history}}\n\nQuestion: {{question}}\n\nAnswer:"
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(llm=llm, output_key="answer", memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, chain_type="stuff", memory=memory,
        verbose=True, return_source_documents=True, combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

st.set_page_config(page_title="Chat with IIT Patna's Chatbot", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    div.css-textbarboxtype {
    background-color: #EEEEEE; 
    border: 1px solid #DCDCDC;
    padding: 5% 5% 5% 10%; 
    border-radius: 10px;
    color: #000000;
    }
    div.css-textbarboxtype:nth-of-type(3) {
        text-align: justify; text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("About Bot")
    st.markdown("## Description")
    st.markdown("""<div class="css-textbarboxtype">An AI-powered chatbot designed to provide answers related to IIT Patna.</div>""", unsafe_allow_html=True)

    st.markdown("## Goals")
    st.markdown("""<div class="css-textbarboxtype">- Student Support<br>- Admissions Guidance<br>- Academic Information<br>- Campus Services</div>""", unsafe_allow_html=True)

    st.markdown("## Purpose")
    st.markdown("""<div class="css-textbarboxtype">Designed as a seamless entry point to IIT Patna's support system.</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Chat History")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if st.button(f"Chat {idx//2 + 1}: {message['content'][:30]}...", key=f"history_{idx}"):
                st.session_state.selected_chat = idx//2

    st.markdown("---")
    if st.button("Export Chat to PDF"):
        if len(st.session_state.chat_history) > 0:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, "IIT Patna Chatbot - Conversation History", ln=True, align='C')
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)

                pdf.set_font('Arial', '', 10)
                for message in st.session_state.chat_history:
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 10, message["role"].capitalize(), ln=True)
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 10, remove_emojis(message["content"]))
                    pdf.ln(5)

                filename = f"iitp_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(filename)

                with open(filename, "rb") as f:
                    st.download_button(label="Download PDF", data=f, file_name=filename, mime="application/pdf")
                os.remove(filename)
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("No chat history to export!")

st.title("🎓 IIT Patna Chatbot")
st.image("https://static.vecteezy.com/system/resources/previews/001/912/491/large_2x/set-of-scenes-business-people-meeting-with-infographics-presentation-free-vector.jpg", use_column_width=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about IIT Patna")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})