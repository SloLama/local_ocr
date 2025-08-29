import pypdfium2 as pdfium
import openai
import os
from io import BytesIO
from tqdm import tqdm
import base64
from argparse import ArgumentParser
import warnings

import torch.multiprocessing as mp

DPI = 192
# TODO: Set to the IP and port of your local server
BASE_URL = "http://localhost:65535/v1"
MODEL_PATH="/models/Llama-4-Maverick-17B-128E-Instruct-FP8"
PROMPT_INSTRUCTION = "Pretvori besedilo na sliki v Markdown tekstovni format. Če slika ne vsebuje besedila, vrni prazen dokument. Pri pretvorbi dosledno upoštevaj naslednji seznam zahtev."
PROMPT_CONDITIONS = [
    "Pazi na ustrezen zapis matematičnih enačb, konstant in spremenljivk. Zapis naj bo v LaTeX formatu, prav tako bodi pozoren, ali je enačba del tesktovne vrstice ali svoja vrstica. Vse matematične enačbe, spremenljivke in konstante naj bodo vsebovane v $, če so del vrstice oz. v $$, če gre za svojo vrstico.",
    "Vrni samo besedilo, ji je na sliki. Ne vračaj ničesar drugega.",
    "Bodi pozoren, kako so besede na sliki zapisane. Tvoj zapis naj bo identičen kot na sliki in ne kakorkoli spremenjen z namenom popravljanja besed.",
    "Pazi na prelome vrstic. Če se ena vrstica konča z znakom '-', to ponavadi pomeni, da je beseda prelomljena čez dve vrstici. Tako prelomljene besede združi v eno enoto, - pa pobriši.",
    "Pazi, da ne vključuješ glav in nog strani. Nekatere strani imajo recimo v glavi napisan naslov knjige oz. revije, v nogi pa številko strani. Te podatke izpusti.",
    "Pazi na formatiranje naslovov. Če se besedilo ne začne z naslovom, ne začni z naslovom. Nekateri naslovi člankov vsebujejo tudi avtorja. Avtorja napiši pod naslov v bold formatu.",
    "Besedilo na sliki je lahko del starinskega časopisa. Bodi pozoren na strukturo besedila."
]


# Function to send an image to Gemini and get a response
def send_image_to_model(image, prompt, client):
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
            {
                "role": "user",
                "content": [
                    image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            max_completion_tokens=4096
        )

        output = response.choices[0].message.content

        return output

    except Exception as e:
        print(f"Error sending image to the model: {e}")
        return None


def get_images_from_pdf(pdf_path, dpi):
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    page_list = [page_to_image(pdf[idx], dpi) for idx in range(num_pages)]

    return [page for page in page_list if page is not None]


def page_to_image(page, dpi):
    try:
        image = page.render(scale=dpi / 72, draw_annots=False).to_pil()
        image = image.convert("RGB")
    except:
        warnings.warn("Skipping PDF page as it is unreadable")
        return None

    # Convert image to bytes for vision model
    image_bytes = BytesIO()
    image.save(image_bytes, format="WEBP")

    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


def process_pdf(args):
    """
    Processes a PDF file, sends each page's image to the local server, and saves the response as Markdown document.

    Args:
        pdf_path (str): The path to the PDF file.
        output_path (str): The path to the directory where each page will be stored as separate MD file.
    """
    pdf_path, output_path, skip_existing = args

    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        if skip_existing:
            return
    else:
        os.makedirs(output_path, exist_ok=True)

    client = model_refs["client"]
    prompt = model_refs["prompt"]

    try:
        page_images = get_images_from_pdf(pdf_path, DPI)

        print(f"Processing document with {len(page_images)} pages")

        for page_num, image in enumerate(page_images):
            if image:
                model_response = send_image_to_model(image, prompt, client)
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
    for condition in PROMPT_CONDITIONS:
        prompt += "\n - " + condition

    client = openai.OpenAI(api_key="test", base_url=BASE_URL)

    return {"client": client, "prompt": prompt}


def main(workers, skip_existing):
    input_dir = "/data"
    output_dir = "/output"

    model_dict = create_model_dict()

    document_list = os.listdir(input_dir)
    n_documents = len(document_list)
    task_args = []
    for i, document in enumerate(document_list):
        document_name = document.removesuffix(".pdf")
        task_args.append((os.path.join(input_dir, document), os.path.join(output_dir, document_name), skip_existing))

    total_processes = min(n_documents, workers)

    with mp.Pool(processes=total_processes, initializer=worker_init, initargs=(model_dict,),
                 maxtasksperchild=None) as pool:
        pbar = tqdm(total=n_documents, desc="Processing PDFs", unit="pdf")
        for _ in pool.imap_unordered(process_pdf, task_args):
            pbar.update(1)
        pbar.close()

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
