import streamlit as st
import logging
import os
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import chromadb
from processing import (
    create_vector_db,
    extract_all_pages_as_images,
    pil_image_to_base64,
    process_question_with_llamaindex,
)
from tts import TextToSpeechService
from stt import SpeechToTextService
import sounddevice as sd  # For playback of TTS audio

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize TTS and STT services
tts_service = TextToSpeechService()
stt_service = SpeechToTextService()

# Initialize PersistentClient for Chroma
persist_dir = "./chroma_data"
os.makedirs(persist_dir, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=persist_dir,
    settings=Settings(),  # Use default settings
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Main Streamlit app function
def main() -> None:
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # File uploader
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", 
        type="pdf", 
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    # Process uploaded PDF
    if file_upload:
        if st.session_state["vector_db"] is None:
            with st.spinner("Processing uploaded PDF..."):
                st.session_state["vector_db"] = create_vector_db(file_upload, chroma_client)
                pdf_pages = extract_all_pages_as_images(file_upload)
                st.session_state["pdf_pages"] = pdf_pages

    # Display PDF pages in a fixed-size, scrollable container
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        with col1:
            pdf_html = """
            <style>
                .pdf-container {
                    height: 600px;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    padding: 10px;
                    background-color: #f9f9f9;
                }
                .pdf-page {
                    margin-bottom: 10px;
                }
            </style>
            <div class="pdf-container">
            """
            for page_image in st.session_state["pdf_pages"]:
                base64_image = pil_image_to_base64(page_image)
                pdf_html += f'<img src="data:image/png;base64,{base64_image}" class="pdf-page" style="width: 100%;"/>'
            pdf_html += "</div>"
            from streamlit.components.v1 import html
            html(pdf_html, height=600)

    # Chat interface
    with col2:
        message_container = st.container()

        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        input_method = st.radio("Input Method", options=["Text", "Speech"], index=0)

        if input_method == "Speech":
            duration = 5
            if st.button("Start Recording"):
                audio_data = stt_service.record_audio(duration)
                transcribed_text = stt_service.transcribe(audio_data)
                st.session_state["messages"].append({"role": "user", "content": transcribed_text})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(transcribed_text)

                if st.session_state["vector_db"] is not None:
                    response = process_question_with_llamaindex(
                        transcribed_text, st.session_state["vector_db"]
                    )
                    with message_container.chat_message("assistant", avatar="🤖"):
                        st.markdown(response)

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )
                    sample_rate, audio_array = tts_service.synthesize(response)
                    sd.play(audio_array, samplerate=sample_rate)

        else:
            if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                if st.session_state["vector_db"] is not None:
                    response = process_question_with_llamaindex(
                        prompt, st.session_state["vector_db"]
                    )
                    with message_container.chat_message("assistant", avatar="🤖"):
                        st.markdown(response)

                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    sample_rate, audio_array = tts_service.synthesize(response)
                    sd.play(audio_array, samplerate=sample_rate)


if __name__ == "__main__":
    main()
