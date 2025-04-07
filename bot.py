# bot.py

import base64
import json
import logging
import mimetypes
import os
from typing import Optional

import requests
from dotenv import load_dotenv

# Use imports based on the lxmfy examples
from lxmfy import LXMFBot, Attachment, AttachmentType

# Define the constant based on attachments.py (adjust if LXMF library provides it)
FIELD_IMAGE = 0x06

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL = "meta-llama/llama-4-maverick"
ASK_MODEL = "meta-llama/llama-4-maverick"
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "llama-4-maverick-bot")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "Llama 4 Maverick Bot")

# Token limits
MAX_INPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 1024
MAX_PROMPT_LENGTH = 2000  # Approximate character limit for prompts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_mime_type(filename: str) -> str:
    """Guess MIME type from filename."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'application/octet-stream'

def encode_image_to_base64(image_data: bytes) -> str:
    """Encode image data to base64 string."""
    return base64.b64encode(image_data).decode('utf-8')

def call_openrouter_vision_api(prompt: str, image_data: bytes, image_mime_type: str) -> Optional[str]:
    """Calls the OpenRouter API with text prompt and image data."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found in environment variables.")
        return "Error: API key is missing."

    base64_image = encode_image_to_base64(image_data)
    data_url = f"data:{image_mime_type};base64,{base64_image}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    }

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt if prompt else "Describe this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": data_url }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
    }

    logger.info(f"Sending request to OpenRouter vision model: {VISION_MODEL} with prompt: '{prompt}'")
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"OpenRouter API Response: {result}")

        if result.get("choices") and len(result["choices"]) > 0:
            message_content = result["choices"][0].get("message", {}).get("content")
            if message_content:
                logger.info("Received successful response from OpenRouter.")
                return message_content.strip()
            else:
                logger.warning("OpenRouter response missing message content.")
                return "Error: Received an empty response from the vision model."
        else:
            error_details = result.get("error", "Unknown error structure")
            logger.warning(f"OpenRouter response indicates an error or no choices: {error_details}")
            return f"Error: Failed to get a valid response from the vision model. Details: {error_details}"

    except requests.exceptions.Timeout:
        logger.error("Request to OpenRouter timed out.")
        return "Error: The request to the vision model timed out."
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Response: {http_err.response.text}")
        return f"Error: Failed to communicate with the vision model (HTTP {http_err.response.status_code})."
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        return "Error: Could not connect to the vision model service."
    except json.JSONDecodeError:
         logger.error(f"Failed to decode JSON response: {response.text}")
         return "Error: Received an invalid response format from the vision model."
    except Exception as e:
        logger.exception("An unexpected error occurred while calling the OpenRouter API.")
        return f"Error: An unexpected error occurred: {e}"

def call_openrouter_ask_api(prompt: str, image_data: bytes, image_mime_type: str) -> Optional[str]:
    """Calls the OpenRouter API with text prompt and image data using the ask model."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found in environment variables.")
        return "Error: API key is missing."

    base64_image = encode_image_to_base64(image_data)
    data_url = f"data:{image_mime_type};base64,{base64_image}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    }

    payload = {
        "model": ASK_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt if prompt else "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": data_url }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
    }

    logger.info(f"Sending request to OpenRouter ask model: {ASK_MODEL} with prompt: '{prompt}'")
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"OpenRouter API Response: {result}")

        if result.get("choices") and len(result["choices"]) > 0:
            message_content = result["choices"][0].get("message", {}).get("content")
            if message_content:
                logger.info("Received successful response from OpenRouter.")
                return message_content.strip()
            else:
                logger.warning("OpenRouter response missing message content.")
                return "Error: Received an empty response from the ask model."
        else:
            error_details = result.get("error", "Unknown error structure")
            logger.warning(f"OpenRouter response indicates an error or no choices: {error_details}")
            return f"Error: Failed to get a valid response from the ask model. Details: {error_details}"

    except requests.exceptions.Timeout:
        logger.error("Request to OpenRouter timed out.")
        return "Error: The request to the ask model timed out."
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Response: {http_err.response.text}")
        return f"Error: Failed to communicate with the ask model (HTTP {http_err.response.status_code})."
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        return "Error: Could not connect to the ask model service."
    except json.JSONDecodeError:
         logger.error(f"Failed to decode JSON response: {response.text}")
         return "Error: Received an invalid response format from the ask model."
    except Exception as e:
        logger.exception("An unexpected error occurred while calling the OpenRouter API.")
        return f"Error: An unexpected error occurred: {e}"

def call_openrouter_text_api(prompt: str) -> Optional[str]:
    """Calls the OpenRouter API with text prompt only."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found in environment variables.")
        return "Error: API key is missing."

    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.warning(f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")
        return f"Error: Your question is too long. Please keep it under {MAX_PROMPT_LENGTH} characters."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    }

    payload = {
        "model": ASK_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    logger.info(f"Sending text request to OpenRouter model: {ASK_MODEL} with prompt: '{prompt}'")
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"OpenRouter API Response: {result}")

        if result.get("choices") and len(result["choices"]) > 0:
            message_content = result["choices"][0].get("message", {}).get("content")
            if message_content:
                logger.info("Received successful response from OpenRouter.")
                return message_content.strip()
            else:
                logger.warning("OpenRouter response missing message content.")
                return "Error: Received an empty response from the model."
        else:
            error_details = result.get("error", "Unknown error structure")
            logger.warning(f"OpenRouter response indicates an error or no choices: {error_details}")
            return f"Error: Failed to get a valid response from the model. Details: {error_details}"

    except requests.exceptions.Timeout:
        logger.error("Request to OpenRouter timed out.")
        return "Error: The request to the model timed out."
    except requests.exceptions.HTTPError as http_err:
        error_response = http_err.response.json() if http_err.response.text else {}
        error_message = error_response.get("error", {}).get("message", "")
        
        if "prompt training" in error_message.lower():
            logger.error("OpenRouter prompt training not enabled")
            return "Error: Prompt training needs to be enabled in your OpenRouter settings. Please visit https://openrouter.ai/settings/privacy to enable it."
        
        logger.error(f"HTTP error occurred: {http_err} - Response: {http_err.response.text}")
        return f"Error: Failed to communicate with the model (HTTP {http_err.response.status_code})."
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        return "Error: Could not connect to the model service."
    except json.JSONDecodeError:
         logger.error(f"Failed to decode JSON response: {response.text}")
         return "Error: Received an invalid response format from the model."
    except Exception as e:
        logger.exception("An unexpected error occurred while calling the OpenRouter API.")
        return f"Error: An unexpected error occurred: {e}"

def extract_first_image_attachment(lxmf_fields: dict) -> Optional[Attachment]:
    """Extracts the first image attachment found in LXMF message fields."""
    if not lxmf_fields:
        return None

    if FIELD_IMAGE in lxmf_fields:
        img_field_data = lxmf_fields[FIELD_IMAGE]
        if isinstance(img_field_data, (list, tuple)) and len(img_field_data) == 2:
            img_format, img_data = img_field_data
            if isinstance(img_format, str) and isinstance(img_data, bytes):
                return Attachment(
                    type=AttachmentType.IMAGE,
                    name=f"image.{img_format}",
                    data=img_data,
                    format=img_format
                )
            else:
                 logger.warning(f"FIELD_IMAGE data has unexpected types: format={type(img_format)}, data={type(img_data)}")
        else:
            logger.warning(f"FIELD_IMAGE data is not a list/tuple of length 2: {img_field_data}")

    logger.debug("No suitable image attachment found in LXMF fields.")
    return None


# --- LXMFy Bot Class ---

class VisionBot:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.bot = LXMFBot(
            name="Llama 4 Maverick Bot",
            command_prefix="", # Or "" if you prefer no prefix
            storage_type="json",
            storage_path="data/llama_4_maverick_bot",
            announce=0,
            first_message_enabled=False
        )
        self.bot.command(name="vision", description="Analyze an attached image with Llama 4 Maverick.")(self.handle_vision_command)
        self.bot.command(name="ask", description="Ask questions to Llama 4 Maverick.")(self.handle_ask_command)

    # Command handler remains synchronous
    def handle_vision_command(self, ctx):
        """Handles the vision command with an image attachment."""
        sender = ctx.sender
        prompt = " ".join(ctx.args) if ctx.args else "Describe this image."

        lxmf_message = ctx.lxmf
        image_attachment = extract_first_image_attachment(getattr(lxmf_message, 'fields', None))

        if not image_attachment:
             logger.warning(f"User {sender} sent vision command without a valid image attachment.")
             # Removed await
             ctx.reply("Please attach an image to your message when using the `vision` command.")
             return

        logger.info(f"Received vision command from {sender} with image: {image_attachment.name} and prompt: '{prompt}'")
        # Removed await
        ctx.reply(f"Analyzing the image '{image_attachment.name}' with prompt: '{prompt}'. Please wait...")

        try:
            image_data = image_attachment.data
            mime_type = get_mime_type(image_attachment.name or "image.png")

            api_response = call_openrouter_vision_api(prompt, image_data, mime_type)

            if api_response:
                max_len = 4000
                for i in range(0, len(api_response), max_len):
                    # Removed await
                    ctx.reply(api_response[i:i+max_len])
            else:
                # Removed await
                ctx.reply("Sorry, I received an unexpected empty response after processing.")

        except Exception as e:
            logger.exception(f"Error processing vision command for {sender}")
            # Removed await
            ctx.reply(f"An unexpected error occurred while processing the image: {e}")

    def handle_ask_command(self, ctx):
        """Handles the ask command with text input."""
        sender = ctx.sender
        prompt = " ".join(ctx.args) if ctx.args else None

        if not prompt:
            logger.warning(f"User {sender} sent ask command without a question.")
            ctx.reply("Please provide a question when using the `ask` command.")
            return

        logger.info(f"Received ask command from {sender} with prompt: '{prompt}'")
        ctx.reply(f"Processing your question: '{prompt}'. Please wait...")

        try:
            api_response = call_openrouter_text_api(prompt)

            if api_response:
                max_len = 4000
                for i in range(0, len(api_response), max_len):
                    ctx.reply(api_response[i:i+max_len])
            else:
                ctx.reply("Sorry, I received an unexpected empty response after processing.")

        except Exception as e:
            logger.exception(f"Error processing ask command for {sender}")
            ctx.reply(f"An unexpected error occurred while processing your question: {e}")

    def run(self):
        """Starts the bot."""
        if not OPENROUTER_API_KEY:
             logger.error("FATAL: OPENROUTER_API_KEY is not set. Please create a .env file with the key.")
             return

        logger.info("Starting Vision Bot...")
        self.bot.run()

# --- Main Execution Block ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the LXMFy Vision Bot.')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable detailed logging.'
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    vision_bot = VisionBot(debug_mode=args.debug)
    vision_bot.run()