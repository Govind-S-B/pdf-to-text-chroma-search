# PDF to Text Chroma Search
Python scripts that converts PDF files to text, splits them into chunks, and stores their vector representations using GPT4All embeddings in a Chroma DB. It also provides a script to query the Chroma DB for similarity search based on user input.

## Requirements

- Python 3.x
- PyPDF2
- chromadb
- langchain

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/pdf-to-text-chroma-search.git
```
2. Install the required dependencies:
```
pip install PyPDF2 chromadb langchain
```

## Usage

### Script 1: Convert PDFs to text, split into chunks, and store in Chroma DB

1. Place your PDF files in the `input` directory.
2. Run the following command to convert the PDFs to text, split them into chunks, and store their vector representations in the Chroma DB:
```
python write_script.py
```

### Script 2: Load Chroma DB and query user input

1. Run the following command to load the Chroma DB and query user input:
```
python read_script.py
```
2. Enter your query when prompted.
