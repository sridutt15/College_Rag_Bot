import os
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_pdf_documents(directory):
    """Load and extract text from PDF files, preserving table layouts."""
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file}...")
            file_path = os.path.join(directory, pdf_file)
            
            # Extract text using pdfplumber with layout preservation
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # layout=True maintains the visual spacing of tables as a text grid!
                    page_text = page.extract_text(layout=True) 
                    if page_text:
                        text += page_text + "\n"
            
            doc = Document(
                page_content=text,
                metadata={"source": pdf_file}
            )
            documents.append(doc)
            print(f"Successfully processed {pdf_file}")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    return documents

def main():
    if not os.path.exists("data"):
        os.makedirs("data")
        return

    if not os.path.exists("vector_db_dir"):
        os.makedirs("vector_db_dir")

    try:
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings()
        
        print("Loading and processing PDF documents...")
        documents = load_pdf_documents("data")
        
        if not documents:
            print("No documents were successfully processed.")
            return
            
        print(f"Successfully loaded {len(documents)} documents")

        # Increased chunk size from 2000 to 2500 to keep full tables intact
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, 
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text_chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(text_chunks)} chunks")

        print("Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )
        
        print("Successfully vectorized and stored documents in 'vector_db_dir'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()