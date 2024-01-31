'''
How to run the code:

- mkdir data
- create a virtual environement:
    python3 -m venv .venv
    . .venv/bin/activate
- pip install langchain pypdf openai chromadb tiktoken docx2txt
- create docs folder and place all of your input files in that
- python {file_name}.py

'''


import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

class DocBot:
    def __init__(self):
        self.documents = []
        self.chat_history = []
        self.initialize_environment()
        self.load_documents()
        self.create_vectordb()
        self.initialize_chat_bot()

    def initialize_environment(self):
        os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

    def load_documents(self):
        for file in os.listdir("docs"):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(f"./docs/{file}")
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(f"./docs/{file}")
            elif file.endswith('.txt'):
                loader = TextLoader(f"./docs/{file}")
            else:
                continue
            self.documents.extend(loader.load())

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.documents = text_splitter.split_documents(self.documents)

    def create_vectordb(self):
        self.vectordb = Chroma.from_documents(self.documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
        self.vectordb.persist()

    def initialize_chat_bot(self):
        self.pdf_qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
            self.vectordb.as_retriever(search_kwargs={'k': 6}),
            return_source_documents=True,
            verbose=False
        )

    def start_chat(self):
        purple = "\033[0;35m"  # ANSI escape code for purple
        blue = "\033[0;34m"    # ANSI escape code for blue
        white = "\033[0;39m"   # ANSI escape code for white



        print(f"{purple} This ChatBot developed by Ethan and Faramarz. We really honered that you chose us \U0001F600")
        print(f"{purple} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


        while True:
            query = input(f"{blue}Prompt: ")
            if query in ["exit", "quit", "q", "f"]:
                print('Exiting')
                sys.exit()

            if query == '':
                continue

            result = self.pdf_qa({"question": query, "chat_history": self.chat_history})
            print(f"{white}Answer: " + result["answer"])
            self.chat_history.append((query, result["answer"]))


if __name__ == "__main__":
    bot = DocBot()
    bot.start_chat()

