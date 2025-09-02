# OCR Slovene PDFs

This repository contains tools and scripts to perform **Optical Character Recognition (OCR)** on **Slovenian PDF documents**, using various methods and models. It is structured into three main components:

- **nanonets_ocr**: OCR using Nanonets-OCR model
- **llama4_ocr**: OCR using a local OpenAI compatible Llama 4 Maverick vLLM server
- **chatgpt_ocr**: OCR using GPT-4o model
- **tesseract_ocr**: Tesseract OCR (+ optional ChatGPT postprocessing)
- **postprocess_markdown**: Classification of OCR-ed Markdown pages and merging them into a single document

---

## Nanonets-OCR

This script uses the [Nanonets-OCR model](https://huggingface.co/nanonets/Nanonets-OCR-s) to perform OCR on PDFs. It processes and saves each page separately as markdown files.

### Usage

To perform the OCR run the `nanonets_ocr/ocr_with_vllm.py` script. The script supports the following command-line arguments:

- `--batch_size`
    - **Type:** `int`  
    - **Required:** Yes  
    - **Default:** None  
    - **Purpose:** Specifies the number of pages to be processed at once by the model.

- `--skip_existing`
  - **Type:** `bool`  
  - **Action:** `store_true`  
  - **Purpose:** If provided, files whose output directories already exist will be skipped during processing.

- `--n_chunks`
  - **Type:** `int`  
  - **Default:** `1`  
  - **Purpose:** Splits the input data into multiple chunks for distributed processing.

- `--chunk_index`
  - **Type:** `int`  
  - **Default:** `0`  
  - **Purpose:** Specifies which chunk of the input data should be processed. Must be an integer between `0` and `n_chunks - 1`.

- `--tp_size`
  - **Type:** `int`  
  - **Default:** `1`  
  - **Purpose:** Determines the number of GPUs across which the model should be split for training or inference using PyTorch's `torch.distributed.launch`.

To run the script on SLURM cluster see SBATCH scripts under `nanonets_ocr`.

---

## Llama 4 Maverick OCR

This code starts a **local OpenAI compatible vLLM server** using [Llama 4 Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) and sends PDF images to it for OCR. Each PDF page is sent separately and the output is stored as separate Markdown file.

### Usage

1. Start the server:

```bash
cd llama4_ocr/server
./run_server.sh meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 <node_ip>
```

If you are running the server and the OCR script on the same node, you can enter `localhost` as `<node_ip>`.

To run the server on a SLURM cluster, see `run_server_script.sbatch` script.

2. Send PDFs to the server for OCR using `ocr_with_local_server.py` script.

#### Arguments

- `--workers`:
  - **Type**: `int`
  - **Required**: Yes
  - **Description**: Specifies the number of documents to convert in parallel. This should be set to the number of CPUs available for optimal performance.
- `--skip_existing`:
  - **Type**: `bool`
  - **Description**: If provided, files whose output directories already exist will be skipped. This helps avoid overwriting existing results.

#### Example Usage

```bash
python llama4_ocr/ocr_with_local_server.py --workers 4 --skip_existing
```
**Important:** Before running the script, set the `BASE_URL` variable in the script to the address of the server, printed by `run_server.sh` script.

To run the OCR on SLURM cluster, see `run_ocr.sbatch` script.

---

## GPT-4o OCR

This script uses OpenAI’s gpt-4o model to perform OCR directly on PDF pages. Each page is rendered at 192 dpi, converted to WEBP, and sent to the API. The outputs are saved as a single text file.

Here's a polished version of your README section for GitHub:

---

### Chatbot OCR on PDF (GPT-4o)

This script uses OpenAI’s [gpt-4o](https://platform.openai.com/gpt-4o) model to perform OCR directly on PDF pages. Each page is rendered at 192 dpi, converted to WEBP format, and sent to the API. The outputs are saved as a single text file with page headers separating the content.

---

### Usage

```bash
python chatgpt_ocr/chatgpt_ocr.py <file.pdf>
```

---

### Arguments

- `--workers`:
  - **Type**: `str`
  - **Required**: Yes
  - **Description**: Path to the PDF file to process.

---

### Output

A single text file named `<file>_chatgpt.txt` is generated, containing:

- All extracted text from each page.
- Page headers included to separate content.

---

### ⚠️ Notes

- **API Key Requirement**: An `API_KEY` variable must be set in the script.
- **Page Processing**: Each page is processed independently.
- **Formatting**: Markdown and other formatting are stripped for evaluation purposes.

---

## **Tesseract OCR**

This Python script performs Optical Character Recognition (OCR) on PDF pages using **Tesseract**, with a specific focus on the **Slovene language**.  Optionally, the recognized text can be sent to **OpenAI's gpt-4o** for correction and improved accuracy.

-----

### **Usage**

To run the script, use the following command in your terminal:

```bash
python tesseract_ocr/tesseract_OCR.py <file.pdf>
```

-----

### **Arguments**

  - `<file.pdf>`
      - **Type**: `str`
      - **Required**: Yes
      - **Description**: The file path to the input PDF document you want to process.

-----

### **Outputs**

The script generates two output files in the same directory as the input PDF:

  - `<file>_raw.txt`
      - Contains the raw text output from the Tesseract OCR process.
  - `<file>_LLM.txt`
      - Contains the OCR text that has been corrected by ChatGPT. This file is only generated if the **ChatGPT postprocessing** is enabled.

-----

### ⚠️ Notes

  - **Tesseract Installation**: You need to have Tesseract installed on your system. Make sure the Slovene language pack (`tesseract-ocr-slv`) is also installed for proper functionality.
  - **Rendering Resolution**: The default rendering resolution for the PDF pages can be adjusted within the script. A resolution of **192 dpi** is generally recommended for optimal results.
  - **API Key**: An **OpenAI API key** is only required if you choose to use the ChatGPT postprocessing feature to correct the OCR text.

## Postprocess Markdown

This code performs following transformations on the OCR output (in Markdown format):

- Classify each page as either content or boilerplate (only content is kept),
- Merges content pages that are part of a single PDF into a single Markdown file.

### Usage

To run the postprocessing, use `postprocess_ocr.py` script. 

#### Arguments

- `--model_path`:
  - **Type**: `str`
  - **Required**: Yes
  - **Description**: Path to the model. Can be a Hugging Face model URI or a local path.

- `--input_dir`:
  - **Type**: `str`
  - **Required**: Yes
  - **Description**: Path to the input directory containing subdirectories that represent separate documents.

- `--output_dir`:
  - **Type**: `str`
  - **Required**: Yes
  - **Description**: Path to the output directory where processed markdown files will be stored.

- `--log_dir`:
  - **Type**: `str`
  - **Required**: Yes
  - **Description**: Path to the log directory where post-processing model predictions are stored.
  
- `--tp_size`:
  - **Type**: `int`
  - **Default**: `1`
  - **Description**: Number of GPUs to use for tensor parallelism.

- `--gpu_memory_utilization`:
  - **Type**: `float`
  - **Default**: `0.8`
  - **Description**: Percentage of GPU memory to use for model weights and KV cache.

- `--batch_size`:
  - **Type**: `int`
  - **Default**: `128`
  - **Description**: Inference batch size.

- `--skip_existing`:
  - **Type**: `bool`
  - **Description**: If provided, files that already exist in the output directory will be skipped.

- `--n_chunks`:
  - **Type**: `int`
  - **Default**: `1`
  - **Description**: Number of chunks to split the input data into for distributed processing.

- `--chunk_index`:
  - **Type**: `int`
  - **Default**: `0`
  - **Description**: Index of the chunk to be processed. Must be between `0` and `n_chunks - 1`.

### Example Usage

```bash
python process_documents.py \
    --model_path "google/gemma-3-27b-it" \
    --input_dir /data/input_dir \
    --output_dir /output \
    --log_dir /log \
    --tp_size 2 \
    --gpu_memory_utilization 0.9 \
    --batch_size 64 \
    --skip_existing \
    --n_chunks 4 \
    --chunk_index 1
```

**NOTE:** The postprocessing script can be run with any vLLM compatible LLM, however we tested it with `google/gemma-3-27b-it` and recommend using this model.

To run the postprocessing on SLURM cluster see SBATCH scripts under `postprocess_markdown`

---

## **License**

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

---

## **Contact**

**Domen Vreš**  
domen.vres@fri.uni-lj.si

---

## **Acknowledgements**

The framework was developed within the following research  research programmes:
- [PoVeJMo](https://www.cjvt.si/povejmo/en/project/) (Adaptive Natural Language Processing with Large Language Models), particularly within the research project titled SloLLaMai -- Open-access computationally efficient models for Slovenian. The program is funded within the Recovery and Resilience Plan by the Slovenian Research and Innovation Agency (ARIS) and NextGenerationEU.
- [LLM4DH](https://www.cjvt.si/llm4dh/en/project/) (Large Language Models for Digital Humanities). The programme is founded by the Slovenian Research and Innovation Agency.

The authors also acknowledge the financial support from the Slovenian Research and Innovation Agency (research core funding No. P6-0411 -- Language Resources and Technologies for Slovene).

---