import pypdfium2 as pdfium
import os
from io import BytesIO
import base64
from argparse import ArgumentParser

from vllm import LLM, SamplingParams

DPI = 192
PROMPT_INSTRUCTION = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
MODEL_PATH = "/models/Nanonets-OCR-s"


# Function to send an image to Gemini and get a response
def send_image_to_model(image, prompt, model, sampling_params):
    """
    Sends an image to the Gemini API and retrieves the generated content.

    Args:
        image (PIL.Image.Image): The image to send to Gemini.
        prompt (str): The text prompt to accompany the image.
        client (openai.OpenAI): OpenAI client containing the vision model
    Returns:
        str: The generated text from Gemini, or None on error.
    """
    try:
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/webp;base64,{}".format(image),
            }
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_data
                ],
            }
        ]

        outputs = model.chat([messages], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        return generated_text

    except Exception as e:
        print(f"Error sending image to the model: {e}")
        return None


def get_images_from_pdf(pdf_path, dpi):
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)

    return [page_to_image(pdf[idx], dpi) for idx in range(num_pages)]


def page_to_image(page, dpi):
    image = page.render(scale=dpi / 72, draw_annots=False).to_pil()
    image = image.convert("RGB")

    # Convert image to bytes for vision model
    image_bytes = BytesIO()
    image.save(image_bytes, format="WEBP")

    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


def process_pdf(args):
    """
    Processes a PDF file, sends each page's image to the local model, and saves the response as Markdown document.

    Args:
        pdf_path (str): The path to the PDF file.
        output_path (str): The path to the directory where each page will be stored as separate MD file.
    """
    pdf_path, output_path, skip_existing, model_refs = args

    if os.path.exists(output_path):
        if skip_existing:
            return
    else:
        os.makedirs(output_path)

    model = model_refs["model"]
    sampling_params = model_refs["sampling_params"]
    prompt = model_refs["prompt"]

    try:
        page_images = get_images_from_pdf(pdf_path, DPI)

        print(f"Processing document with {len(page_images)} pages")

        for page_num, image in enumerate(page_images):
            if image:
                model_response = send_image_to_model(image, prompt, model, sampling_params)
                with open(os.path.join(output_path, f"page_{page_num + 1}.md"), "w") as f_out:
                    f_out.write(model_response)
            else:
                print(f"Failed to convert page {page_num + 1} to image.")

    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def worker_init(model_dict):
    if model_dict is None:
        model_dict = create_model_dict()

    global model_refs
    model_refs = model_dict


def create_model_dict():
    prompt = PROMPT_INSTRUCTION
    model = LLM(MODEL_PATH, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32000)

    return {"model": model, "sampling_params": sampling_params, "prompt": prompt}


def main(skip_existing):
    input_dir = "/data"
    output_dir = "/output"

    document_list = os.listdir(input_dir)

    model_refs = create_model_dict()

    for i, document in enumerate(document_list):
        document_name = document.removesuffix(".pdf")
        task_args = (
            os.path.join(input_dir, document), os.path.join(output_dir, document_name),
            skip_existing,
            model_refs
        )
        process_pdf(task_args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--skip_existing", action="store_true",
                        help="If given, files whose output dirs exists will be skipped.")
    args = parser.parse_args()
    main(args.skip_existing)
