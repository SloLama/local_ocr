import os
import sys
import base64
import requests
from io import BytesIO
import pypdfium2 as pdfium

# Set your OpenAI API key here or use an environment variable
API_KEY = ''


def get_images_from_pdf(pdf_path, dpi=192):
    pdf = pdfium.PdfDocument(pdf_path)
    return [page_to_image(pdf[i], dpi) for i in range(len(pdf))]

def page_to_image(page, dpi):
    image = page.render(scale=dpi / 72, draw_annots=False).to_pil()
    image = image.convert("RGB")
    image_bytes = BytesIO()
    image.save(image_bytes, format="WEBP")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

def ocr_page(image_base64, page_num):
    print(f"Processing page {page_num + 1}...")
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Please extract the text (OCR) from this scanned page."},
                {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{image_base64}"}}
            ]}
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return content

def main():
    if len(sys.argv) != 2:
        print("Usage: python ocr_script.py <file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    output_path = os.path.splitext(pdf_path)[0] + "_chatgpt.txt"
    images_base64 = get_images_from_pdf(pdf_path)
    full_text = []

    for i, image_b64 in enumerate(images_base64):
        try:
            text = ocr_page(image_b64, i)
            full_text.append(f"--- Page {i + 1} ---\n{text}\n")
        except Exception as e:
            print(f"Error processing page {i + 1}: {e}")
            full_text.append(f"--- Page {i + 1} ---\n[Error processing page]\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

    print(f"OCR completed. Output saved to {output_path}")

if __name__ == "__main__":
    main()

