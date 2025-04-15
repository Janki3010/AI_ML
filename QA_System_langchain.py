from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # Load your OpenAI API key from .env

# Step 1: Load Text
text = """
Harry Potter is a young wizard who discovers his magical heritage on his 11th birthday. He attends Hogwarts School of Witchcraft and Wizardry, where he learns about friendship, bravery, and the dark wizard Voldemort who killed his parents. Harry becomes a hero by confronting evil, aided by friends Ron and Hermione.
"""

# Step 2: Convert to LangChain Document
documents = [Document(page_content=text)]

# Step 3: Split Document
splitter = SentenceTransformersTokenTextSplitter(model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_overlap=0, tokens_per_chunk=100)
docs_split = splitter.split_documents(documents)

# Step 4: Create Embeddings and VectorStore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(docs_split, embedding_model)

# Step 5: Set up the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Step 6: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=False
)

# Step 7: Gradio Interface
def ask(question):
    result = qa_chain.run(question)
    return result

gr_interface = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, placeholder="Ask something about Harry Potter..."),
    outputs="text"
)

gr_interface.launch()
