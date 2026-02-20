import gradio as gr
import pandas as pd
import os
from prophet import Prophet
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from groq import Groq

# Load Prediction Model
def predict_inflation():
    df = pd.read_csv("inflation.csv")
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=1, freq='Y')
    forecast = model.predict(future)

    return round(forecast['yhat'].iloc[-1], 2)

# Load Knowledge Base
def load_vectorstore():
    with open("knowledge_base/inflation_causes.txt", "r") as f:
        text = f.read()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore

vectorstore = load_vectorstore()

# Load Groq API
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chatbot(user_input):
    if "predict" in user_input.lower():
        prediction = predict_inflation()
        return f"Predicted Inflation Next Year: {prediction}%"

    docs = vectorstore.similarity_search(user_input)
    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
    Use this context to answer:

    {context}

    Question: {user_input}
    """

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="EcoMind AI - Inflation Advisor"
)

iface.launch()
