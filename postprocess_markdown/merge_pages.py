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


def process_too_long_prompt(prompt):
    prompt_input = prompt.split("Vhod:<zadnji_odstavek>")[-1].split("</zadnji_odstavek>\n<prvi_odstavek>")
    last_paragraph = prompt_input[0]
    first_paragraph = prompt_input[1][:-len("</prvi_odstavek>")]

    if len(last_paragraph) > len(first_paragraph):
        prediction = f"<združeno_besedilo>{first_paragraph}</združeno_besedilo>\n"
        prediction += "<razlaga>Prevelika dolžina prompta. Posledično je zadnji odstavek prve strani odstranjen, saj je daljši.</razlaga>"
    else:
        prediction = f"<združeno_besedilo>{last_paragraph}</združeno_besedilo>\n"
        prediction += "<razlaga>Prevelika dolžina prompta. Posledično je prvi odstavek druge strani odstranjen, saj je daljši.</razlaga>"

    return prediction


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
                prediction = process_too_long_prompt(prompts[i])
                predictions.append(prediction)
                warnings.warn("Page merging avoided due to prompt being too long.")
            else:
                prediction = response.__next__()
                predictions.append(prediction.outputs[0].text)

    return predictions


def process_batch(batch, model, sampling_params, prompt_template, tokenizer):
    prompts = []
    for example in batch:
        prompt = prompt_template.replace("{last_paragraph}", example["last_paragraph"])
        prompt = prompt.replace("{first_paragraph}", example["first_paragraph"])
        prompts.append(prompt)

    predictions = generate_responses(prompts, model, sampling_params, tokenizer)

    return predictions


def get_number_from_file(filename):
    number_part = filename.split("_")[1]
    number = number_part.split(".")[0]

    return int(number)


def create_example(current_page, next_page):
    return {"last_paragraph": current_page[-1], "first_paragraph": next_page[0]}


def merge_pages(page_split, merge_predictions):
    # Add first paragraph of the document
    first_page = page_split[0][0]
    if len(first_page) > 1:
        merged_document = [first_page[0]]
    else:
        merged_document = []
    merging_unit = None

    for ((current_page, next_page), prediction) in zip(page_split, merge_predictions):
        # If we have only one paragraph we need to check whether it is already included
        if len(current_page) == 1:
            if merging_unit is None:
                merging_unit = prediction
            else:
                merging_unit += prediction.removeprefix(current_page[0])
        else:
            if len(current_page) > 2:
                merged_document += current_page[1:-1]

            merging_unit = prediction

        if len(next_page) == 1:
            if not merging_unit.endswith(next_page[0]):
                merged_document += [merging_unit]
                merging_unit = None
        elif merging_unit is not None:
            merged_document += [merging_unit]
            merging_unit = None

    # Add the last page
    if merging_unit is not None:
        merged_document += [merging_unit]
    last_page = page_split[-1][1]
    if len(last_page) > 1:
        merged_document += last_page[1:]

    return "\n\n".join(merged_document)


def process_document(input_pages, output_file, model, sampling_params, batch_size, tokenizer):
    if len(input_pages) == 0:
        print("No pages for merging. Return None.")
        return None

    if len(input_pages) == 1:
        print("Skipping page merging as there is only 1 page.")
        return input_pages[0]

    prompt_instruction = "Podana sta dva odstavka - zadnji odstavek trenutne strani (<zadnji_odstavek>) in prvi odstavek naslednje strani (<prvi_odstavek>). Tvoja naloga je združiti ta dva odstavka v celoto. Pri združevanju bodi pozoren tako na format kot vsebino obeh odstavkov in **ne dodajaj, ne spreminjaj ali ne briši nobene druge vsebine razen tiste, ki je nujna za pravilno združitev odstavkov**."

    prompt_properties = "Pri združevanju upoštevaj naslednji seznam navodil in kriterijev:"
    properties = [
        "Ali zadnji odstavek trenutne strani predstavlja nogo strani - recimo številko strani ali kakšne druge informacije nepomembne za vsebino strani? V tem primeru v združeni celoti ta odstavek odstrani (odstavka ne vključi v združeno besedilo). Pri tem pazi, da ne odstranjuješ opomb ob koncu strani, npr. '$^{82}$ Andrej Studen, \"Bojimo se, da bo tekla kri!\", Zgodovina za vse, leto II, št. 2, Celje, 1995, str. 2.'.",
        "Ali prvi odstavek naslednje strani predstavlja glavo strani - recimo informacije o časopisu, avtorju, založbi? V tem primeru  v združeni celoti ta odstavek odstrani (odstavka ne vključi v združeno besedilo). Pri tem pazi, da ne odstranjuješ naslovov in avtorjev poglavij, npr. '# MED ZMAGO IN PORAZOM\n## Narodni dom med svetovnima vojnama'.",
        "Ali se zadnji odstavek trenutne strani zaključi s prelomom besede, ki se nadaljuje v prvem odstavku naslednje strani? Prelome besed identificiraj s pomočjo znaka '-'. V tem primeru odstavka združi tako, da prelomljeno besedo združiš v eno brez znaka '-'.",
        "Ali je mogoče odstavka združiti v en odstavek? Pri tem upoštevaj dolžino in vsebinsko povezanost odstavkov - če gre za kratka odstavka, ki sta vsebinsko povezana, gre za isti odstavek. V tem primeru odstavka združi brez preloma vrstice.",
        "Ali odstavka predstavljata različna odstavka? Recimo, da gre za dva daljša odstavka ali dva odstavka, ki se med sabo vsebinsko ne povezujeta. V tem primeru odstavka združi z dvema prelomoma vrstice, saj ne gre za isto enoto.",
    ]
    for i, property in enumerate(properties):
        prompt_properties += f"\n{i + 1}. {property}"

    prompt_examples = "Pomagaj si z naslednjimi rešenimi primeri."
    examples = [
        "Primer, ko se odstrani zadnji odstavek trenutne strani:\n<zadnji_odstavek>Informacije: 01 839-11-17 ali 01 832-53-12 popoldne (18h)\nZaoj nas najdete tudi na internetu: www.altera.si ali nam pišete:\nalteria@email.si in se prijavite. Srečali se tako, tudi kar tako, vpisate.\nVISE PRIJAVLJENE BOMO 0 PRVIH ŠESTKIH OBVESTILI PO TEL. ALI PIŠNO!!</zadnji_odstavek>\n<prvi_odstavek>## VRTNARSTVO GAŠPERLIN\nVrtnaška vzgoja in trgovina  \nMošnje 99 pri Komendi, tel.: 83-43-587  \n- SADIKE VRTNIH JAGOD  \n- ČEBULICE  \n- RAZNE TRAJNICE  \n- SOBNE ROŽE,  \n- GNOJILA, SUBSTRATI...  \nObišćite nas od 8. do 17. ure,  \nob sobotah od 8. do 13. ure!  \nNaše cene so v vrtu.</prvi_odstavek>\n<združeno_besedilo>## VRTNARSTVO GAŠPERLIN\nVrtnaška vzgoja in trgovina  \nMošnje 99 pri Komendi, tel.: 83-43-587  \n- SADIKE VRTNIH JAGOD  \n- ČEBULICE  \n- RAZNE TRAJNICE  \n- SOBNE ROŽE,  \n- GNOJILA, SUBSTRATI...  \nObišćite nas od 8. do 17. ure,  \nob sobotah od 8. do 13. ure!  \nNaše cene so v vrtu.</združeno_besedilo>",
        "Primer, ko se odstrani prvi odstavek naslednje strani:\n<zadnji_odstavek>$^{4}$ Janez Cvirn, *Boj za Celje*, Ljubljana, 1989, str. 21, 37, 60.</zadnji_odstavek>\n<prvi_odstavek>Članki in razprave - Articles and treatises                                 Celjski zbornik 1997</prvi_odstavek>\n<združeno_besedilo>$^{4}$ Janez Cvirn, *Boj za Celje*, Ljubljana, 1989, str. 21, 37, 60.</združeno_besedilo>",
        "Primer, ko se odstavka združita preko preloma besede:\n<zadnji_odstavek>Pri drugi rešitvi uporabljajo tako imenovano optično izravnavo. Rotirajoče zrcalo pošilja svetlobo do gorilnika s frekvenco okrog 10 na sek. izmenoma skozi celico, ki vsebuje preiskovano snov, in praznno celico ali polnjene samo s topilom. Oba žarka nadaljujeta isto pot skozi monohro-mator in zadevata izmenoma na sprejemnik. Če se celici razlikujeta po prepustnosti, to je, če snov pri določeni valovni dolžini absorbira, bo na sprejemniku nastala izmenična napetost s frekvenco rotirjočega zrcala in amplitudo, ki je sorazmerna razliki prepustnosti. Ta napetost se ojačuje in preko servomehanizmov prenaša na premakljivo zaslonko, ki slabi žarek na strani prazne celice. Slabitev je proporcionalna premiku zaslonke. Za-takrat namreč na sprejemniku ne nastaja več izmenična napetost. Z zaslonko je povezano pero, ki beleži premike zaslonke kot odstopke absorpci-je. Smer gibanja zaslonke in z njo povezane peresa je odvisna od faze napetosti na sprejemniku, faza pa od strani, s katere prihaja več energije. Ob prehodu kakšnega absorpcijskega pasu se zaslonka zapira do izravnave, ko obstane. Pri zmanjšanju absorpci-je pa se faza napetosti na sprejemniku obrne, s čimer se obrne tudi smer gibanja zaslonke, ki se začne zopet odpirati. Opisani sistem zahteva visoko kvaliteto optičnih delov in zelo na-</zadnji_odstavek>\n<prvi_odstavek>tančno uravnano, ker se morata slikati obeh žarkov, ki izmenoma padata na sprejemnik, popolnoma kriti. Zelo občutljiv del je tudi zaslonka, od katere je odvisna točnost meritve prepustnosti. Zato pa je instrument precej neodvisen od sprememb okolice med snemanjem. V prodaji je več modelov, ki so približno polovicno cenejši od prej opisanega (ca. 12 000 dolarjev).</prvi_odstavek>\n<združeno_besedilo>Pri drugi rešitvi uporabljajo tako imenovano optično izravnavo. Rotirajoče zrcalo pošilja svetlobo do gorilnika s frekvenco okrog 10 na sek. izmenoma skozi celico, ki vsebuje preiskovano snov, in praznno celico ali polnjene samo s topilom. Oba žarka nadaljujeta isto pot skozi monohro-mator in zadevata izmenoma na sprejemnik. Če se celici razlikujeta po prepustnosti, to je, če snov pri določeni valovni dolžini absorbira, bo na sprejemniku nastala izmenična napetost s frekvenco rotirjočega zrcala in amplitudo, ki je sorazmerna razliki prepustnosti. Ta napetost se ojačuje in preko servomehanizmov prenaša na premakljivo zaslonko, ki slabi žarek na strani prazne celice. Slabitev je proporcionalna premiku zaslonke. Za-takrat namreč na sprejemniku ne nastaja več izmenična napetost. Z zaslonko je povezano pero, ki beleži premike zaslonke kot odstopke absorpci-je. Smer gibanja zaslonke in z njo povezane peresa je odvisna od faze napetosti na sprejemniku, faza pa od strani, s katere prihaja več energije. Ob prehodu kakšnega absorpcijskega pasu se zaslonka zapira do izravnave, ko obstane. Pri zmanjšanju absorpci-je pa se faza napetosti na sprejemniku obrne, s čimer se obrne tudi smer gibanja zaslonke, ki se začne zopet odpirati. Opisani sistem zahteva visoko kvaliteto optičnih delov in zelo natančno uravnano, ker se morata slikati obeh žarkov, ki izmenoma padata na sprejemnik, popolnoma kriti. Zelo občutljiv del je tudi zaslonka, od katere je odvisna točnost meritve prepustnosti. Zato pa je instrument precej neodvisen od sprememb okolice med snemanjem. V prodaji je več modelov, ki so približno polovicno cenejši od prej opisanega (ca. 12 000 dolarjev).</združeno_besedilo>",
        "Primer, ko se odstavka združita v enega:\n<zadnji_odstavek>Zaporedje distribucij $T_{1}, T_{2}, T_{3}, \ldots$ konvergira proti distribuciji $T$, ako je za poljubno funkcijo $\varphi$ iz razreda $(D)$ vedno $T_{n}(\varphi) \to T(\varphi)$. Iz te</zadnji_odstavek>\n<prvi_odstavek>definicije je takoJ razvidno, da se sme vsako konvergentno zaporedje distribucij členoma odvajati. Če je torej $T_n \to T$, velja tudi $T'_n \to T'$. Vsa pravila so za distribucije veliko bolj preprosta in enotna kakor za navadne funkcije.</prvi_odstavek>\n<združeno_besedilo>Zaporedje distribucij $T_{1}, T_{2}, T_{3}, \ldots$ konvergira proti distribuciji $T$, ako je za poljubno funkcijo $\varphi$ iz razreda $(D)$ vedno $T_{n}(\varphi) \to T(\varphi)$. Iz te definicije je takoJ razvidno, da se sme vsako konvergentno zaporedje distribucij členoma odvajati. Če je torej $T_n \to T$, velja tudi $T'_n \to T'$. Vsa pravila so za distribucije veliko bolj preprosta in enotna kakor za navadne funkcije.</združeno_besedilo>",
        "Primer, ko odstavka predstavljata različni enoti:\n<zadnji_odstavek>Pridobljene pravice in ugodnosti pa so posredno vplivale tudi na discipliniranje delojemalcev, saj so nevtralizirale potencialno konfliktne teme, tako da ni prihajalo do socialno-političnih artikulacij razrednega nasprotja, ki tiči v statusu delojemalca. Torej cilj koncepta (in projekta) socialne države ni bil ukiniti razredno nasprotje oziroma revolucionirati heteronomno delo, pač pa le čim bolj preprečiti rizike, ki so izhajali iz delovnih razmerij.</zadnji_odstavek>\n<prvi_odstavek>Socialna država je lahko polno zaposlenost kot svoj temeljni cilj dosegala le posredno. Na voljo je imela zakonodajno moč, ki ji je omogočala sprejemanje regulativnih mehanizmov, s katerimi je lahko oblikovala pogoje za uresničevanje njenega temeljnega cilja. Njena prva naloga je tako bila, da oblikuje komu-nikacijski prostor za kolektivno dogovarjanje med seboj neodvisnih in organiziranih subjektov (delodajalcev in sindikatov), ki so neposredno odločali o doseganju polne zaposlenosti.</prvi_odstavek>\n<združeno_besedilo>Pridobljene pravice in ugodnosti pa so posredno vplivale tudi na discipliniranje delojemalcev, saj so nevtralizirale potencialno konfliktne teme, tako da ni prihajalo do socialno-političnih artikulacij razrednega nasprotja, ki tiči v statusu delojemalca. Torej cilj koncepta (in projekta) socialne države ni bil ukiniti razredno nasprotje oziroma revolucionirati heteronomno delo, pač pa le čim bolj preprečiti rizike, ki so izhajali iz delovnih razmerij.\n\nSocialna država je lahko polno zaposlenost kot svoj temeljni cilj dosegala le posredno. Na voljo je imela zakonodajno moč, ki ji je omogočala sprejemanje regulativnih mehanizmov, s katerimi je lahko oblikovala pogoje za uresničevanje njenega temeljnega cilja. Njena prva naloga je tako bila, da oblikuje komu-nikacijski prostor za kolektivno dogovarjanje med seboj neodvisnih in organiziranih subjektov (delodajalcev in sindikatov), ki so neposredno odločali o doseganju polne zaposlenosti.</združeno_besedilo>"
    ]
    for example in examples:
        prompt_examples += "\n\n" + example

    prompt_format = "Vrni združena odstavka ter kratko razlago tvoje odločitve. Pri združevanju pazi, da ne spremeniš obeh odstavkov na noben drug način kot zgolj za združevanje. Na vhodu bo zadnji odstavek trenutne strani ozčen z `<zadnji_odstavek>`, prvi odstavek naslednje strani pa z `<prvi_odstavek>`. Tvoj izhod mora biti v naslednji obliki:\n`<združeno_besedilo>ZDRUŽENA ODSTAVKA</združeno_besedilo>\n<razlaga>KRATKA RAZLAGA</razlaga>`"
    prompt_template = f"{prompt_instruction}\n\n{prompt_properties}\n\n{prompt_examples}\n\n{prompt_format}\n\nVhod:<zadnji_odstavek>" + "{last_paragraph}</zadnji_odstavek>\n<prvi_odstavek>{first_paragraph}</prvi_odstavek>"

    pages = [page.split("\n\n") for page in input_pages]
    inputs = [create_example(current_page, next_page) for current_page, next_page in zip(pages, pages[1:])]

    f_out = open(output_file, "a")

    n_batches = ceil(len(inputs) / batch_size)
    merge_predictions = []
    for i in trange(n_batches):
        input_batch = inputs[i * batch_size: min(len(pages), (i + 1) * batch_size)]
        batch_results = process_batch(
            input_batch,
            model,
            sampling_params,
            prompt_template,
            tokenizer
        )

        for input_example, prediction in zip(input_batch, batch_results):
            extracted_prediction = extract_content_from_tag(prediction, "združeno_besedilo")
            if extracted_prediction is None:
                extracted_prediction = input_example["last_paragraph"] + "\n\n" + input_example["first_paragraph"]
            merge_predictions.append(extracted_prediction)
            input_example["prediction"] = prediction
            f_out.write(json.dumps(input_example, ensure_ascii=False) + "\n")

    f_out.close()

    merged_document = merge_pages(list(zip(pages, pages[1:])), merge_predictions)

    return merged_document


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
        max_tokens=4096
    )

    page_files = os.listdir(args.input_dir)
    page_files.sort(key=get_number_from_file)
    pages = []
    for file in page_files:
        with open(os.path.join(args.input_dir, file)) as f_in:
            pages.append(f_in.read())

    process_document(pages, args.output_file, model, sampling_params, args.batch_size, tokenizer)

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
