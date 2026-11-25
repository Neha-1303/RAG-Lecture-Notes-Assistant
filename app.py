"""Gradio interface for the RAG Lecture-Notes Assistant."""
from pathlib import Path
import gradio as gr
from config import DATA_DIR
from ingest import ingest_folder
from index import build_index
from rag import RAGAssistant

assistant = None

def build_or_refresh_index(files):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_count = 0
    for file_obj in files or []:
        # Gradio File(type="filepath") supplies file paths as strings.
        source = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
        destination = DATA_DIR / source.name
        if source.resolve() != destination.resolve():
            destination.write_bytes(source.read_bytes())
        uploaded_count += 1
    try:
        records = ingest_folder(DATA_DIR)
        count = build_index(records)
        global assistant
        assistant = None  # force reload against the refreshed index
        return f"Index successfully built: {count} chunks from the PDFs in data/."
    except Exception as exc:
        return f"Could not build index: {exc}"

def respond(message, history):
    global assistant
    if not message or not message.strip():
        return "Please enter a question."
    try:
        if assistant is None:
            assistant = RAGAssistant()
        return assistant.answer(message, history)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return (
            f"An error occurred: {exc}\n\n"
            "Check your .env settings, API provider configuration, internet connection, "
            "and whether the FAISS index has been built."
        )

with gr.Blocks(title="RAG Lecture-Notes Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# RAG Lecture-Notes Assistant\n"
        "Ask questions from your own PDF lecture notes. Answers are grounded in retrieved "
        "passages and include source filenames and page numbers."
    )
    with gr.Accordion("Upload lecture notes / rebuild index", open=True):
        pdf_upload = gr.File(
            label="Select PDF lecture notes",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath",
        )
        build_button = gr.Button("Build / refresh index", variant="primary")
        index_status = gr.Markdown("Add PDFs, then build the index before chatting.")
        build_button.click(
            fn=build_or_refresh_index,
            inputs=[pdf_upload],
            outputs=[index_status],
        )
    gr.Markdown("## Ask your notes")
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=480),
        textbox=gr.Textbox(
            placeholder="Example: Explain the characteristics of cloud computing from my notes...",
            lines=2
        ),
        examples=[
            "Summarise the main points from my lecture notes.",
            "Explain the most important concept in simple language.",
            "Which page discusses the topic I asked about?",
        ],
        title="Lecture Notes Chat",
    )
    gr.Markdown(
        "**Note:** The assistant may refuse if the indexed notes do not contain enough "
        "evidence. Always verify important points against the original PDF."
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
