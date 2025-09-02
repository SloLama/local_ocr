import fitz  # PyMuPDF
import requests
from PIL import Image
import io
import base64
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path

# Your OpenAI API Key
API_KEY = ''


def extract_page_as_image(pdf_path, page_index, resolution=192):
    """
    Extracts a specific page from a PDF as an image and saves it as PNG.

    Parameters:
        pdf_path (str): Path to the PDF file.
        page_index (int): Index of the page to extract (0-based).
        resolution (int): DPI resolution for the PNG.

    Returns:
        PIL.Image: Image object of the extracted page.
    """

    images = convert_from_path(pdf_path, dpi=resolution)  # Increase dpi for quality
    images[page_index].save("output_pdf2image.png", "PNG")

    return images[page_index]


def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.adaptiveThreshold(cv2.medianBlur(gray, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return Image.fromarray(gray)


def perform_ocr(image, language="slv"):
    """
    Performs OCR on a PIL Image using Tesseract.

    Parameters:
        image (PIL.Image): The image to perform OCR on.

    Returns:
        str: OCR text extracted from the image.
    """

    return pytesseract.image_to_string(image, lang=language)


def send_text_to_chatgpt(text, api_key):
    """
    Sends OCR text to ChatGPT API for correction and enhancement.

    Parameters:
        text (str): The OCR-extracted text to correct.
        api_key (str): OpenAI API key.

    Returns:
        str: Enhanced or corrected text from ChatGPT.
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    ocr_correction_prompt = f"""Correct OCR-induced errors in the text written in Slovene language (a historical text). Follow these guidelines:
1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Remove unnecessary line breaks within sentences or paragraphs to ensure smooth reading flow.
   - Preserve meaningful paragraph breaks as they appear in the text.
   - Combine fragmented lines into full sentences where appropriate.
   - Fix common OCR errors (e.g., 'rn' misread as 'm')
   - Use context and linguistic knowledge to fix errors, but do not make speculative changes.
   - Focus only on clear errors; do not modify valid content unnecessarily.
   - Do not add extra periods or any unnecessary punctuation unless required by grammatical correctness.

2. Maintain original structure:
   - Preserve all headings, subheadings, and their formatting.
   - Do not merge or split paragraphs unless required to fix clear formatting issues caused by OCR.

3. Preserve original content:
   - Keep all important information from the original text unchanged.
   - Do not add, infer, or introduce any new information.

4. Maintain coherence:
   - Handle incomplete sentences gracefully: Correct partial sentences to make them grammatically and contextually correct and Resolve any disruptions caused by OCR errors that may fragment sentences or ideas.

IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks, except where fixing unnecessary line breaks within sentences or paragraphs. Do not include any introduction, explanation, or metadata.

Original Text in Slovene:
{text}"""

    data = {
        "model": "gpt-4o",  # or "gpt-3.5-turbo" chatgpt-4o-latest
        "messages": [
            {"role": "system", "content": "You are an expert in correcting OCR errors."},
            ##{"role": "user", "content": f"Please correct any OCR errors in the following historical Slovene text:\n\n{text}"}
            {"role": "user", "content": ocr_correction_prompt}
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    print(response)

    return response.json()['choices'][0]['message']['content']


if __name__ == "__main__":
    if __name__ == "__main__":
        if len(sys.argv) != 2:
            print("Usage: python tesseract_ocr.py <file.pdf>")
            sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    filename_result_raw = f"{pdf_path.rsplit('.', 1)[0]}_raw.txt"
    filename_result_LLM = f"{pdf_path.rsplit('.', 1)[0]}_LLM.txt"

    # Entire file extraction mode:

    # Initialize variables to store combined text
    combined_raw_text = ""
    combined_corrected_text = ""

    # Open the PDF to get the number of pages
    with fitz.open(pdf_path) as pdf:
        num_pages = pdf.page_count

    # Iterate through all pages in the PDF
    for page_index in range(num_pages):
        print(f"Processing page {page_index + 1}/{num_pages}...")

        try:
            # Step 1: Extract the specified page as an image
            print("Extracting page from pdf as image")
            image = extract_page_as_image(pdf_path, page_index, 192)

            image = preprocess_image(image)

            # Step 2: Perform OCR on the extracted image
            print("Performing Tesseract OCR on the extracted image")
            ocr_text = perform_ocr(image)
            print(f"OCR Text for page {page_index + 1}:", ocr_text[:200])  # Show snippet of OCR output

            # Step 3: Send the OCR text to ChatGPT for correction
            print("Enhancing and formatting OCR output using ChatGPT")
            corrected_text = send_text_to_chatgpt(ocr_text, API_KEY)
            print(f"Corrected Text for page {page_index + 1}:",
                  corrected_text[:200])  # Show snippet of corrected output

            # Save results to the files immediately
            with open(filename_result_raw, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== Page {page_index + 1} ===\n\n{ocr_text}")

            with open(filename_result_LLM, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== Page {page_index + 1} ===\n\n{corrected_text}")

            print(f"Page {page_index + 1} processed successfully.")

        except Exception as e:
            print(f"Error processing page {page_index + 1}: {e}")
            continue  # Skip to the next page in case of an error

    print(f"Processing complete. Results saved to '{filename_result_raw}' and '{filename_result_LLM}'.")
