import streamlit as st
import logging
import os
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import chromadb
from toolcalling import chatbot_response
from processing import (
    create_vector_db,
    extract_all_pages_as_images,
    pil_image_to_base64,
    process_question_with_llamaindex,
)
from tts import TextToSpeechService
from stt import SpeechToTextService
import sounddevice as sd  # For playback of TTS audio
from generateimage import generate_image
from describeimage import describe_image  # Import describe_image function

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF/Image RAG Streamlit UI",
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
    st.subheader("🧠 Ollama PDF/Image RAG playground", divider="gray", anchor=False)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # File uploader
    file_upload = col1.file_uploader(
        "Upload a PDF or Image ↓", 
        type=["pdf", "jpg", "jpeg", "png"], 
        accept_multiple_files=False,
        key="file_uploader"
    )
    file_upload_type = None
    # Display uploaded file (PDF or image)
    if file_upload:
        file_type = file_upload.type
        file_upload_type = "pdf"
        if file_type == "application/pdf":
            # Process PDF
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["vector_db"] = create_vector_db(file_upload, chroma_client)
                    pdf_pages = extract_all_pages_as_images(file_upload)
                    st.session_state["pdf_pages"] = pdf_pages
            with col1:
            # Display PDF pages
                if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
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

        elif file_type in ["image/jpeg", "image/png"]:
            file_upload_type = "image"
            # Display image
            col1.image(file_upload, caption="Uploaded Image", use_column_width=True)

    # Chat interface and input
    with col2:
        message_container = st.container()

        # Display previous messages
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Text input or speech handling
        input_method = st.radio("Input Method", ["Text", "Speech"], index=0)
        
        # Speech input
        if not file_upload:  # Chỉ gọi chatbot_response nếu không có file PDF được tải lên
            if input_method == "Speech":
                duration = 5
                if st.button("Start Recording"):
                    # Ghi âm và xử lý đầu vào từ người dùng
                    audio_data = stt_service.record_audio(duration)
                    transcribed_text = stt_service.transcribe(audio_data)
                    st.session_state["messages"].append({"role": "user", "content": transcribed_text})
                    with message_container.chat_message("user", avatar="😎"):
                        st.markdown(transcribed_text)

                    # if "draw" in transcribed_text.lower():
                    #     st.write("Detected 'draw' in the speech. Generating image...")
                    #     try:
                    #         image_url = generate_image(transcribed_text)
                    #         st.image(image_url, caption=f"Generated Image: {transcribed_text}", use_column_width=True)
                    #     except Exception as e:
                    #         st.error(f"Error generating image: {e}")
                    # else:
                    bot_response = chatbot_response(transcribed_text)
                    with message_container.chat_message("assistant", avatar="🤖"):
                        st.markdown(bot_response)
                    st.session_state["messages"].append({"role": "assistant", "content": bot_response})

            else:  # Xử lý đầu vào dạng Text
                if prompt := st.chat_input("Enter a prompt here..."):
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    with message_container.chat_message("user", avatar="😎"):
                        st.markdown(prompt)

                    # if "draw" in prompt.lower():
                    #     st.write("Detected 'draw' in the prompt. Generating image...")
                    #     try:
                    #         image_url = generate_image(prompt)
                    #         st.image(image_url, caption=f"Generated Image: {prompt}", use_container_width=True)
                    #     except Exception as e:
                    #         st.error(f"Error generating image: {e}")
                    # else:
                    bot_response = chatbot_response(prompt)
                    with message_container.chat_message("assistant", avatar="🤖"):
                        st.markdown(bot_response)
                    st.session_state["messages"].append({"role": "assistant", "content": bot_response})

        elif file_upload and input_method == "Text":  # Khi có file được tải lên
            if prompt := st.chat_input("Enter a prompt related to the uploaded file..."):
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # if "draw" in prompt.lower():
                #     st.write("Detected 'draw' in the prompt. Generating image...")
                #     try:
                #         image_url = generate_image(prompt)
                #         st.image(image_url, caption=f"Generated Image: {prompt}", use_column_width=True)
                #     except Exception as e:
                #         st.error(f"Error generating image: {e}")
                
                if file_upload_type == "pdf" and st.session_state["vector_db"]:  # Nếu file là PDF
                    response = process_question_with_llamaindex(prompt, st.session_state["vector_db"])
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    with message_container.chat_message("assistant", avatar="🤖"):
                        st.markdown(response)
                        # message_index = len(st.session_state['messages']) - 1
                        # if st.button("🔊 Nghe phản hồi", key=f"listen_{message_index}"):
                        #     sample_rate, audio_array = tts_service.synthesize(response)
                        #     sd.play(audio_array, samplerate=sample_rate)
                    sample_rate, audio_array = tts_service.synthesize(response)
                    sd.play(audio_array, samplerate=sample_rate)

                elif file_upload_type == "image":  # Nếu file là ảnh
                    with st.spinner("Describing the uploaded image..."):
                        try:
                            image_description = describe_image(file_upload, prompt)
                            st.session_state["messages"].append({"role": "assistant", "content": image_description})
                            with message_container.chat_message("assistant", avatar="🤖"):
                                st.markdown(image_description)
                                # if st.button("🔊 Nghe phản hồi", key=f"listen_{len(st.session_state['messages'])}"):
                                #     sample_rate, audio_array = tts_service.synthesize(image_description)
                                #     sd.play(audio_array, samplerate=sample_rate)
                        except Exception as e:
                            st.error(f"Error describing image: {e}")

# def show_image(image):
#     st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
