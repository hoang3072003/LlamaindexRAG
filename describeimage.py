import base64
import google.generativeai as genai

# Configure Generative AI
genai.configure(api_key="AIzaSyCVSFxhcx5B-b4pmW1Ywy1TEoK1xOGTbjg")

# Choose a Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

def describe_image(image_file, user_prompt):
    """
    Describe the content of an image using Gemini AI.

    Args:
        image_file: The uploaded image file object.
        user_prompt: The user-provided prompt for the image.

    Returns:
        str: Description or content generated for the image.
    """
    try:
        # Read the image file
        image_bytes = image_file.read()

        # Call the Gemini model with the image and prompt
        response = model.generate_content(
            [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                },
                user_prompt,
            ]
        )

        # Return the response text
        return response.text
    except Exception as e:
        raise RuntimeError(f"An error occurred while describing the image: {e}")
