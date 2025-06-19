import pypdfium2 as pdfium
import os
from io import BytesIO
import base64
from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

import torch

DPI = 192
PROMPT_INSTRUCTION = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
MODEL_PATH = "/models/Nanonets-OCR-s"


# Function to send an image to Gemini and get a response
def send_image_to_model(image_tuple, prompt, model, processor):
    """
    Sends an image to the Gemini API and retrieves the generated content.

    Args:
        image (PIL.Image.Image): The image to send to Gemini.
        prompt (str): The text prompt to accompany the image.
        client (openai.OpenAI): OpenAI client containing the vision model
    Returns:
        str: The generated text from Gemini, or None on error.
    """

    image_base64, image_pil = image_tuple

    try:
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/webp;base64,{}".format(image_base64),
            }
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print("Text:", text)
        inputs = processor(text=[text], images=[image_pil], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=15000, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print("Output text:", output_text)
        return output_text[0]

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

    return base64.b64encode(image_bytes.getvalue()).decode("utf-8"), image


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
    processor = model_refs["processor"]
    prompt = model_refs["prompt"]

    try:
        page_images = get_images_from_pdf(pdf_path, DPI)

        print(f"Processing document with {len(page_images)} pages")

        for page_num, image in enumerate(page_images):
            if image:
                model_response = send_image_to_model(image, prompt, model, processor)
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

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    return {"model": model, "processor": processor, "prompt": prompt}


def main(workers, skip_existing):
    input_dir = "/data"
    output_dir = "/output"

    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA visible devices:", os.environ["CUDA_VISIBLE_DEVICES"])

    document_list = os.listdir(input_dir)
    n_documents = len(document_list)
    task_args = []

    model_refs = create_model_dict()

    for i, document in enumerate(document_list):
        document_name = document.removesuffix(".pdf")
        task_args = (
        os.path.join(input_dir, document), os.path.join(output_dir, document_name), skip_existing, model_refs)
        process_pdf(task_args)

    """for i, document in enumerate(document_list):
        document_name = document.removesuffix(".pdf")
        task_args.append((os.path.join(input_dir, document), os.path.join(output_dir, document_name), skip_existing))

    total_processes = min(n_documents, workers)

    with mp.Pool(processes=total_processes, initializer=worker_init, initargs=(model_dict,), maxtasksperchild=None) as pool:
        pbar = tqdm(total=n_documents, desc="Processing PDFs", unit="pdf")
        for _ in pool.imap_unordered(process_pdf, task_args):
            pbar.update(1)
        pbar.close()"""

    # Delete the client
    del model_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers", type=int, required=True,
                        help="Number of documents to convert in parallel. Set to number of CPUs.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="If given, files whose output dirs exists will be skipped.")
    args = parser.parse_args()
    main(args.workers, args.skip_existing)
