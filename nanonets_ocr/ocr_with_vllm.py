import pypdfium2 as pdfium
import os
from io import BytesIO
import base64
from argparse import ArgumentParser

from vllm import LLM, SamplingParams

DPI = 192
PROMPT_INSTRUCTION = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
MODEL_PATH = "/models/Nanonets-OCR-s"


class BatchPredictor:
    def __init__(self, batch_size):
        self.model = LLM(MODEL_PATH)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=16000)
        self.batch = {"output_files": [], "messages": []}

        self.batch_size = batch_size

    def add_to_batch(self, message, output_file):
        self.batch["messages"].append(message)
        self.batch["output_files"].append(output_file)

        if len(self.batch["output_files"]) == self.batch_size:
            self.predict_on_batch()

    def predict_on_batch(self):
        if self.batch["output_files"] == []:
            return

        outputs = self.model.chat(
            self.batch["messages"],
            sampling_params=self.sampling_params
        )

        for output_file, output in zip(self.batch["output_files"], outputs):
            with open(output_file, "w") as f_out:
                f_out.write(output.outputs[0].text)

        self.batch = {"output_files": [], "messages": []}


def get_page_prompt(image):
    """
    Creates the prompt for the model, that is added to the current batch

    Args:
        image: base64 encoded image
    Returns:
        list: The input conversation for the model
    """
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
                {"type": "text", "text": PROMPT_INSTRUCTION},
                image_data
            ],
        }
    ]

    return messages


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
    pdf_path, output_path, skip_existing, batch_predictor = args

    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        if skip_existing:
            return
    else:
        os.makedirs(output_path, exist_ok=True)

    try:
        page_images = get_images_from_pdf(pdf_path, DPI)

        print(f"Processing document with {len(page_images)} pages")

        for page_num, image in enumerate(page_images):
            if image:
                prompt_conversation = get_page_prompt(image)
                output_file = os.path.join(output_path, f"page_{page_num + 1}.md")
                batch_predictor.add_to_batch(prompt_conversation, output_file)
            else:
                print(f"Failed to convert page {page_num + 1} to image.")

    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main(batch_size, skip_existing):
    input_dir = "/data"
    output_dir = "/output"

    document_list = os.listdir(input_dir)
    n_documents = len(document_list)

    batch_predictor = BatchPredictor(batch_size)

    for i, document in enumerate(document_list):
        print(f"Processing document {i + 1} of {n_documents}")

        document_name = document.removesuffix(".pdf")
        task_args = (
        os.path.join(input_dir, document), os.path.join(output_dir, document_name), skip_existing, batch_predictor)
        process_pdf(task_args)

    # Process last batch, that can be incomplete
    batch_predictor.predict_on_batch()

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Number of pages to be processed at once by the model.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="If given, files whose output dirs exists will be skipped.")
    args = parser.parse_args()
    main(args.batch_size, args.skip_existing)
