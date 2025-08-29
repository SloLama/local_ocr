from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import trange
from argparse import ArgumentParser
from math import ceil
import json
import os
import warnings

from utils import extract_content_from_tag

MAX_MODEL_LEN = 32768
TOO_LONG_PREDICTION = "<razred>ostalo</razred>\n<razlaga>Stran je daljša od maksimalne dovoljene dolžine in je avtomatsko klasificirana kot šum.</razlaga>\n<stopnja>0</stopnja>"


def generate_responses(prompts, model, sampling_params, tokenizer):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    try:
        response = model.chat(messages, sampling_params, use_tqdm=False)
        predictions = [x.outputs[0].text for x in response]
    # Catch the error when prompt is longer than the model length
    except Exception as e:
        tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        problematic_messages = [idx for idx, tokens in enumerate(tokenized) if len(tokens) > MAX_MODEL_LEN]
        # In case some other error ocurred
        if len(problematic_messages) == 0:
            raise e

        messages = [message for idx, message in enumerate(messages) if idx not in problematic_messages]
        response = iter(model.chat(messages, sampling_params, use_tqdm=False))
        predictions = []
        for i in range(len(prompts)):
            if i in problematic_messages:
                predictions.append(TOO_LONG_PREDICTION)
                warnings.warn("Page avoided due to being too long")
            else:
                prediction = response.__next__()
                predictions.append(prediction.outputs[0].text)

    return predictions


def process_batch(batch, model, sampling_params, prompt_template, tokenizer):
    prompts = [prompt_template.replace("{input}", page) for page in batch]

    predictions = generate_responses(prompts, model, sampling_params, tokenizer)

    return predictions


def get_number_from_file(filename):
    number_part = filename.split("_")[1]
    number = number_part.split(".")[0]

    return int(number)


def process_document(input_dir, output_file, model, sampling_params, batch_size, tokenizer):
    prompt_instruction = "Tvoja naloga je klasificirati podano besedilo kot 'vsebina' ali 'ostalo'. 'Vsebina' se nanaša na strani knjig, člankov ali časopisov, ki predstavljajo glavno pripoved, argumentacijo ali informacije. Pod vsebino spadajo tudi časopisni izseki, poročila ter druga besedila, ki vsebujejo informacije. 'Ostalo' zajema vse druge vrste strani."
    prompt_properties = "Razmisli o naslednjih značilnostih besedila:"
    properties = [
        "Ali besedilo vsebuje smiselne stavke in odstavke, ki tvorijo koherentno celoto?",
        "Ali se informacije v besedilu logično nadaljujejo?",
        "Ali besedilo obravnava določeno temo ali podaja informacije o njej?"
        "Ali besedilo spominja na besedilo iz revije, časopisa, knjige, literarnega dela, znanstvenega dela, bloga, poročila, sporočila, izjave, govora, eseja?"
    ]
    for property in properties:
        prompt_properties += "\n-" + property
    prompt_boilerplate = "Nekaj primerov, ki se klasificirajo kot 'ostalo':"
    boilerplate_list = [
        "Naslovnica knjige oz. časopisa",
        "Kratek vhod, ki vsebujejo zgolj naslove del oz. poglavij",
        "Podatki o založbi",
        "Stran s kazalom",
        "Stran, ki vsebuje pretežno nesmiselne znake, na primer '@@@ ### %%% &&& ***'",
        "Informacije o knjigi",
        "Podatki o knjižnici"
    ]
    for example in boilerplate_list:
        prompt_boilerplate += "\n- " + example
    prompt_format = "Vrni klasifikacijo podanega vhoda, razlog za tvojo odločitev ter stopnjo prepričanja v točnost odgovora. Stopnja naj bo število med 1 in 5. Vhodni tekst bo označen z `<vhod>`. Tvoj izhod mora biti v naslednji obliki:\n`<razred>KLASIFIKACIJA</razred>\n<razlaga>KRATKA RAZLAGA</razlaga>\n<stopnja>STOPNJA OD 1 DO 5</stopnja>`"
    prompt_template = f"{prompt_instruction}\n\n{prompt_properties}\n\n{prompt_boilerplate}\n\n{prompt_format}\n\n<vhod>\n" + "{input}\n</vhod>"

    page_files = os.listdir(input_dir)
    page_files.sort(key=get_number_from_file)
    pages = []
    for file in page_files:
        with open(os.path.join(input_dir, file)) as f_in:
            pages.append(f_in.read())

    if len(pages) == 1:
        print("Skipping classification as there is only 1 page.")
        return pages

    f_out = open(output_file, "a")

    n_batches = ceil(len(pages) / batch_size)
    labels = []
    for i in trange(n_batches):
        batch_results = process_batch(
            pages[i * batch_size: min(len(pages), (i + 1) * batch_size)],
            model,
            sampling_params,
            prompt_template,
            tokenizer
        )

        batch_files = page_files[i * batch_size: min(len(pages), (i + 1) * batch_size)]
        for page, prediction in zip(batch_files, batch_results):
            extracted_label = extract_content_from_tag(prediction, "razred")
            if extracted_label is None:
                extracted_label = prediction.strip()
            labels.append(extracted_label)
            example = {"page": page, "prediction": prediction}
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")

    f_out.close()

    filtered_pages = [page for (page, label) in zip(pages, labels) if label.lower() != "ostalo"]

    return filtered_pages


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (either HF URI or local path)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input text file, where each line is one lemma."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the JSONL file with outputs."
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Number of GPUs to split model across using tensor parallelism."
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="Percentage of GPU memory to use for model weights and KV cache."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Inference batch size."
    )
    return parser.parse_args()


def main(args):
    print("Loading model ...")
    model = LLM(
        args.model_path,
        seed=42,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024
    )

    process_document(args.input_dir, args.output_file, model, sampling_params, args.batch_size, tokenizer)

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
