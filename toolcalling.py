import google.generativeai as genai

# Configure Generative AI
genai.configure(api_key="AIzaSyBZ4VaW3Rw5wPYqU8nttVeziqHFogYSK_E")

# Choose a Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

def chatbot_response(user_input):
    """
    Generate a chatbot response using Gemini AI model.
    
    Args:
        user_input (str): Input query from the user.

    Returns:
        str: AI-generated response.
    """
    # Tạo prompt cho Gemini với các chức năng mới
    prompt = f"""
    You are a helpful assistant. Answer the user's question below briefly. 
    Additionally, suggest available services they can use for their query and guide them on how to use these services.

    Available services:
    1. QA from document: Prompt the user to upload a document, and you will extract and answer questions from it.
    2. Generate Image: Ask the user to describe what they want to draw. Use the keyword "draw" to initiate the image generation process.
    3. Describe an image: Prompt the user to upload an image, and you will analyze and describe the content of the image.

    User: {user_input}
    Bot:
    """

    # Call Gemini API
    response = model.generate_content(prompt)
    final_response = response.text.strip()

    return final_response
