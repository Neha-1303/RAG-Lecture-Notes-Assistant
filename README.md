# RAG Lecture-Notes Assistant

A browser-based question-answering assistant that answers from your own PDF lecture notes using Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF lecture notes through the Gradio interface.
- Extract text page by page and preserve source filename and page number.
- Split extracted text into overlapping chunks.
- Create normalized sentence embeddings with `all-MiniLM-L6-v2`.
- Store vectors in a local FAISS inner-product index.
- Retrieve the most relevant passages for each question.
- Generate answers from retrieved passages only through an OpenAI-compatible hosted LLM API.
- Display source citations and the retrieved passages.
- Return an explicit “not found in the notes” response when retrieval evidence is insufficient.
- Use recent chat history to rewrite short follow-up questions before retrieval.

## Architecture

```text
PDF lecture notes
      |
      v
ingest.py  -- extract text, page metadata, overlapping chunks
      |
      v
index.py   -- Sentence-Transformers embeddings --> FAISS index + metadata
                                                   |
Student question --> retriever.py -----------------+
                         |
                         v
                    rag.py -- grounded prompt --> Hosted LLM API
                         |
                         v
                    app.py (Gradio chat)
                         |
                         v
             Answer + filename/page citations + excerpts
```

## Requirements

- Python 3.10 or later
- Internet connection for downloading the embedding model and calling the LLM API
- An API key for an OpenAI-compatible chat-completions endpoint
- PDF files containing selectable text. Scanned PDFs need OCR before indexing.

## Setup (Windows PowerShell)

```powershell
# 1. Extract the ZIP and enter the project folder
cd RAG-Lecture-Notes-Assistant

# 2. Create and activate a virtual environment
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Create your environment file
Copy-Item .env.example .env
```

Open `.env` and set your API key and model. Do not commit `.env`.

Example configuration (use the values supplied by your chosen API provider):

```env
LLM_API_KEY=your_real_api_key
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=your-chat-model
```

The endpoint must support the OpenAI-compatible `/chat/completions` API. `LLM_BASE_URL` should be the provider's API base URL, usually ending in `/v1`.

## Add your lecture notes

Place your PDF files inside the `data/` folder. You can also upload PDFs from the web interface after starting the app.

## Build / rebuild the FAISS index

```powershell
python ingest.py
```

The index and metadata are saved in `storage/`. Re-run this command whenever you add, remove, or replace PDFs in `data/`.

## Run the application

```powershell
python app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

You can also upload PDFs in the interface. After upload, click **Build / refresh index** before asking questions.

## Important configuration

The defaults are intentionally small enough for a personal course corpus:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: 350 words
- Chunk overlap: 60 words
- Top-k: 4
- Minimum similarity: 0.25
- Temperature: 0.1

Tune these values using a labelled evaluation set rather than assuming they are optimal.

## Evaluation

The report proposes these test groups:

| Test group | Expected behaviour |
|---|---|
| Direct factual | Retrieves the correct page and cites it |
| Paraphrased | Retrieves relevant text despite different wording |
| Multi-section | Combines relevant excerpts and cites both |
| Out-of-scope | Says the notes do not contain the answer |
| Instruction override | Does not follow conflicting instructions in notes or question |

Suggested measures:

- Retrieval hit rate at k
- Mean reciprocal rank (MRR)
- Answer faithfulness
- Correct-refusal rate
- Instruction-override resistance
- Query latency and API failure rate

No accuracy or faithfulness score is claimed by this starter implementation. Evaluate it on a labelled question set from your own notes.

## GitHub upload

Create an empty repository on GitHub, then run:

```powershell
git init
git add .
git commit -m "Initial RAG Lecture-Notes Assistant project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your repository URL.

### Before pushing

- Confirm `.env` is not staged: `git status`
- Never upload API keys, private lecture notes, or generated indexes unless you intentionally want them public.
- The `.gitignore` excludes `.env`, PDFs in `data/`, and generated files in `storage/`.

## Project structure

```text
RAG-Lecture-Notes-Assistant/
├── app.py
├── ingest.py
├── index.py
├── retriever.py
├── rag.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── .gitkeep
└── storage/
    └── .gitkeep
```

## Limitations

- Retrieval quality limits answer quality.
- Text extraction from complex tables, diagrams and equations may be imperfect.
- Scanned PDFs require OCR.
- Prompt-level protections reduce but do not guarantee resistance to prompt injection.
- The hosted LLM requires network access and may apply rate limits.
- This prototype has no user authentication or audit logging.
- Uploaded PDFs are processed locally; retrieved text is sent to the configured LLM provider for generation. Check that provider's data policy before using confidential material.
