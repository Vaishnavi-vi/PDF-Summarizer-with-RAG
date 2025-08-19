from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from PIL import Image
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config("Pdf Summarizer",layout="centered")

page=st.sidebar.radio("Select one between the two:",["Intro","Summarizer"])
if page=="Intro":
    st.header("Pdf Summarizer with RAG....")
    image=Image.open("C:\\Users\\Dell\\Downloads\\Summarizer.png")
    st.image(image,use_container_width=True)
    
elif page=="Summarizer":
    st.header("Welcome the Pdf summarizer App")
    file=st.file_uploader("Add Your pdf here",type="pdf")
    if st.button("Summary"):
        if file==None:
            st.warning("Please add pdf first")
        else:
            with st.spinner("Thinking..."): 
                st.success("Here is your summary")
            #save and load file temporarily
                with open("temp.pdf","wb") as f:
                     f.write(file.read())
            #document loader
                loader=PyPDFLoader("temp.pdf")
                docs=loader.load()
            #text_splitter
                text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20)
                chunks=text_splitter.split_documents(docs)
            #embedding
                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            #vector store
                vector_store=Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name="my_collections")
            #retriever
                retriever=vector_store.as_retriever()
            #model
                llm1=HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",model_kwargs={"api_key":os.getenv("HUGGINGFACE_API_TOKEN")},temperature=0.5,max_new_tokens=512)
                model=ChatHuggingFace(llm=llm1)
            #query
                query="Give the summary of the file."
            #retriver
                retriver_qa=RetrievalQA.from_chain_type(
                llm=model,
                retriever=retriever,
                return_source_documents=True)
                output=retriver_qa.invoke(query)
                st.write(output["result"])
            
    
