from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack import Pipeline

import gradio as gr

# Step 1: Load Your Custom Text (e.g., Harry Potter summary)
text = """
Harry Potter is a young wizard who discovers his magical heritage on his 11th birthday. He attends Hogwarts School of Witchcraft and Wizardry, where he learns about friendship, bravery, and the dark wizard Voldemort who killed his parents. Harry becomes a hero by confronting evil, aided by friends Ron and Hermione.
"""

# Step 2: Initialize In-Memory Document Store
document_store = InMemoryDocumentStore()

# Step 3: Turn the text into Document
doc = Document(content=text)
splitter = DocumentSplitter(split_by="sentence", split_length=2)
splitter.warm_up()
split_docs = splitter.run([doc])["documents"]
document_store.write_documents(split_docs, policy="overwrite")

# Step 4: Embed the Documents
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
embedded_docs = doc_embedder.run(split_docs)["documents"]
document_store.write_documents(embedded_docs, policy="overwrite")

# Step 5: Create Retriever and Text Embedder
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Step 6: Prompt Template for Generator
template = """
Answer the question based on the context below.

Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=template)

# Step 7: Generator (OpenAI)
from dotenv import load_dotenv
from haystack.components.generators import OpenAIGenerator

load_dotenv()

chat_generator = OpenAIGenerator(model="gpt-4o")

#Step 8: Build Pipline
pipeline = Pipeline()

# Add components to your pipline
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", chat_generator)

# Now, connect the components to each other
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.prompt")

def ask(question):
    result = pipeline.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question}
    })
    return result["llm"]["replies"][0]

gr_interface = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text"
)
gr_interface.launch()

# question = "Who are Harry Potter's friends?"
# answer = ask(question)
# print("Q:", question)
# print("A:", answer)

