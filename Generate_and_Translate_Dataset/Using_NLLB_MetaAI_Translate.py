'''
TranNhiem 2023/05/12 

Following Feature to Run inference NLLB MetaAI Translate model as High Performance:

1. Converted the main function to an asynchronous function (async def main(start, end, subset)).
2. Used the asyncio.gather function to run the translate_text_nllb function in parallel for all three columns (instruction, input, output) of the input data.
3. Modified the translation process to preprocess each text before translating it using the preprocess_text function.
4. Created separate lists for translated instructions, inputs, and outputs.
5. Stored the translations in a DataFrame (translations_df) with the appropriate column names.
6. Saved the translations to a JSON file using the save_translated_subset_to_json function.

## Note: NLLB MetaAI Translate model is not available in HuggingFace model hub, so we need to download it from the NLLB MetaAI Translate model repository on GitHub and load it using the AutoModelForSeq2SeqLM class.
https://huggingface.co/facebook/nllb-200-distilled-1.3B 
'''

import pandas as pd
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time 
import os 
import torch
import json
from processing_text import preprocess_text, is_translatable, load_input_data, load_input_data_index


## Save the translated subset to a JSON file
def save_translated_subset_to_json(translated_subset_df, file_path):
    """
    Save the translated subset DataFrame to a JSON file.

    Args:
        translated_subset_df (pd.DataFrame): Translated subset as a DataFrame.
        file_path (str): Output file path.
    """
    translated_subset_dict = translated_subset_df.to_dict('records')
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(translated_subset_dict, outfile, ensure_ascii=False)

weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/NLLB/"
#Create the weight_path if it is not exist 
if not os.path.exists(weight_path): 
    os.makedirs(weight_path)

#Initialize the tokenizer and model
#Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B",cache_dir=weight_path)
model = model.to(device)
source_language={
    "🇱🇷 English": "eng_Latn",
    "🇻🇳 Vietnamese": "vie_Latn", 
    "TraditionalChinese": "zho_Hant",
    "🇨🇳 SimplifiedChinese": "zho_Hans",
    "🇫🇷 French" : "fra_Latn",
    "🇩🇪 German": "deu_Latn",
    "🇲🇨 Indonesian": "ind_Latn",
    "🇯🇵 Japanese": "jpn_Jpan",
    "🇰🇷 Korean": "kor_Hang", 
    "🇪🇸 Spanish": "spa_Latn", 
    "🇹🇭 Thai": "tha_Thai",
    "": "empty",
}
## Translation function
async def translate_text_nllb(text, source_language, target_language):
    """
    Translate the text using the NLLB model.

    Args:
        text (str): Input text.
        source_language (str): Source language code.
        target_language (str): Target language code.

    Returns:
        str: Translated text.
    """
    # Checking whether the text is translatable or not
    if not is_translatable(text):
        return text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=600)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
        max_length=800,
        early_stopping=True
    )

    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

# Main function
async def main(start, end, subset):
    """
    Main function to perform translation.

    Args:
        start (int): Start index of the data subset.
        end (int): End index of the data subset.
        subset (bool): Whether to use a subset of the data.

    Returns:
        None
    """
    input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")

    # Subset the data if needed
    if subset:
        input_data = input_data.iloc[start:end]

    df_length = len(input_data)
    print(f"The length of the dataframe is: {df_length}")

 
    # Translate in parallel
    translations = await asyncio.gather(
        *[translate_text_nllb(preprocess_text(text), source_language['🇱🇷 English'], source_language['🇻🇳 Vietnamese']) for text in input_data['instruction']],
        *[translate_text_nllb(preprocess_text(text), source_language['🇱🇷 English'], source_language['🇻🇳 Vietnamese']) for text in input_data['input']],
        *[translate_text_nllb(preprocess_text(text), source_language['🇱🇷 English'], source_language['🇻🇳 Vietnamese']) for text in input_data['output']]
            )

    # Store the translations in separate lists
    translated_instructions = translations[:len(input_data)]
    translated_inputs = translations[len(input_data):2*len(input_data)]
    translated_outputs = translations[2*len(input_data):]
    
    # Store the translations in a DataFrame
    translations_df = pd.DataFrame({
        "instruction": translated_instructions,
        "input": translated_inputs,
        "output": translated_outputs
    })

    # Save the translations to a JSON file
    save_translated_subset_to_json(translations_df, file_path="./data/output/NLLB_translations_TraditionalChinese_40_51k76.json")


#--------------------------------------------------------------
## Inference Model with Batch -- Under Development 
#--------------------------------------------------------------
# Translation function
# async def translate_batch_nllb(texts, source_language, target_language):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=600)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     translated_tokens = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
#         max_length=800,
#         early_stopping=True
#     )

#     translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
#     return translated_texts
# # Main function

# async def main(start, end, subset):
#     input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")

#     # Subset the data if needed
#     if subset:
#         input_data = input_data.iloc[start:end]

#     df_length = len(input_data)
#     print(f"The length of the dataframe is: {df_length}")

#     # # Initialize the tokenizer and model
#     # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
#     # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
#     # model = model.to(device)

#     # Translate in parallel
#     batch_size = 8
#     translations = []
#     for i in range(0, len(input_data), batch_size):
#         batch_texts = input_data.iloc[i:i + batch_size]['instruction'].tolist()
#         translated_batch = await translate_batch_nllb(batch_texts, source_language['🇱🇷 English'], source_language['🇻🇳 Vietnamese'], )
#         translations.extend(translated_batch)

#     # Store the translations in separate lists
#     translated_instructions = translations[:df_length]
#     translated_inputs = translations[df_length:2 * df_length]
#     translated_outputs = translations[2 * df_length:3 * df_length]

#     # Store the translations in a DataFrame
#     translations_df = pd.DataFrame({
#         "instruction": translated_instructions,
#         "input": translated_inputs,
#         "output": translated_outputs
#     })

#     # Save the translations to a JSON file
#     save_translated_subset_to_json(translations_df, file_path="./data/output/NLLB_translations_Vietnamese_test_1.json")



# Run the asyncio event loop
start = 40000
end = 51760
subset = True
start_time = time.time()
loop = asyncio.get_event_loop()
## BLOOMZ Model
#loop.run_until_complete(main_bloomz(start, end, subset))
## NLLB Model 
loop.run_until_complete(main(start, end, subset))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")