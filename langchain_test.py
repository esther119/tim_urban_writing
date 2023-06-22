# %%
f = open("sample.txt", "r")
data = f.read().replace('\n', '')

# %%
data

# %%
openai_api_key = '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'

# Helper to read local files
import os

# Vector Support
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Model and chain
from langchain.chat_models import ChatOpenAI

# Text splitters
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613', openai_api_key=openai_api_key)

# %%
embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=openai_api_key)

# %%
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

# %%

from langchain.document_loaders import TextLoader
loader = TextLoader('./sample.txt')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 50,
    length_function = len)

docs = text_splitter.split_documents(documents) # list of lists of strings

db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")



# %%
# Now let's test it out
query = "What is first principles?"
docs = db.similarity_search(query)
store = ''
for doc in docs:
    print(doc.page_content)
    store += doc.page_content


# %%
# Response style 1
from langchain.chains import RetrievalQA

retriever = db.as_retriever()

# Conversational QA retrieval chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# template = '''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.'''
# template = '''Use the following pieces of context to answer the question at the end. If you don't know the answer, just clarify that you don't know, but still generate new answers based on the tone of the text.'''
template = ''
user_input = '''Write a post about apocolypse.'''
final_input = template+ store+ user_input
response = qa.run(final_input)
print(f"AI: {response}")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break

#     response = qa.run(user_input)
#     print(f"AI: {response}")


# %%
# Response style 2
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="refine")

# %%
response = chain({"input_documents": docs, "question": user_input, "language": "English", "existing_answer" : store}, return_only_outputs=True)


# %%
import pprint
pprint.pprint(response)


# %%



