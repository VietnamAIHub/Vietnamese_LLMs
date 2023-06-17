'''
@@TranNhiem 2023/06/13

This Code using to Clean the Translation Output from the GPT-3.5-turbo API (OpenAI) & GPT-4 API (OpenAI) for the Vietnamese Language

    The translation might Existing serveral Issue as follow: 
    1. The output might be Replicated 
    2. The output might existing double quotation marks
    3. The output might have False Translation
    4.The output might have Note from the translation with English to explain the Vietnamese word


'''

import json
import os
import re

def merge_json_files():
    """
    Merge multiple JSON files into a single JSON file.
    Checking rows with 'instruction' and 'output' keys are exist in the JSON file.
    """
    merged_data = []

    # Find JSON files with names matching the pattern
    pattern = r'.*checkpoint_(\d+)\.json'
    json_directory = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/0_10k_alpaca/"

    file_list = os.listdir(json_directory)
    filtered_files = sorted([file for file in file_list if re.match(pattern, file)], key=lambda x: int(re.match(pattern, x).group(1)))

    for file_name in filtered_files:
        file_path = os.path.join(json_directory, file_name)
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                for item in data:
                    if "instruction" in item and "output" in item:
                        instruction = item.get("instruction", "")
                        input_data = item.get("input", "")
                        output = item.get("output", "")

                        if isinstance(instruction, list):
                            instruction = instruction[0]

                        if isinstance(input_data, list):
                            input_data = input_data[0]

                        if isinstance(output, list):
                            output = output[0]

                        new_entry = {
                            'instruction': instruction,
                            'input': input_data,
                            'output': output
                        }

                        # Check if the new entry already exists in merged_data
                        if new_entry not in merged_data:
                            merged_data.append(new_entry)

            except json.JSONDecodeError as e:
                print(f"Error parsing file: {file_path}. Error message: {str(e)}")

    # Write merged data to a new file
    output_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_10_20k.json"
    with open(output_file_path, 'w') as file:
        json.dump(merged_data, file)

    print(f"Merged data has been written to {output_file_path}")

# Call the function to merge the JSON files
merge_json_files()

def remove_extra_quotation_marks(json_file_path, output_file_path):
    """
    Remove extra quotation marks from string values in a JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    for item in data:
        for key in item:
            value = item[key]
            if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                item[key] = value[1:-1]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

    print(f"Modified JSON file has been written to {output_file_path}")

# Specify the input JSON file path and the output file path for the modified JSON
input_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_1_10k.json"
output_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_1_10k.json"

# Call the function to remove extra quotation marks
remove_extra_quotation_marks(input_file_path, output_file_path)

def remove_failed_rows(json_file_path, output_file_path):
    """
    Remove rows containing "TRANSLATION_FAILED" from a JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Remove rows containing "TRANSLATION_FAILED"
    data = [item for item in data if not any(value is not None and "TRANSLATION_FAILED" in value for value in item.values())]

    with open(output_file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Modified JSON file has been written to {output_file_path}")

# Specify the input JSON file path and the output file path for the modified JSON
input_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_10_20k.json"
output_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_10_20k.json"

# Call the function to remove failed rows
remove_failed_rows(input_file_path, output_file_path)

def remove_note_from_values(json_file_path, output_file_path):
    """
    Remove "Note: " followed by any English words from the values in each row of a JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Remove "Note: " followed by any English words from the values in each row
    pattern = r'Note: [A-Za-z\s]+'
    for item in data:
        for key, value in item.items():
            if isinstance(value, str):
                item[key] = re.sub(pattern, '', value)

    with open(output_file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Modified JSON file has been written to {output_file_path}")

# Specify the input JSON file path and the output file path for the modified JSON
input_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_10_20k.json"
output_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_10_20k.json"

# Call the function to remove the specified pattern from the values
remove_note_from_values(input_file_path, output_file_path)

def print_examples(json_file_path, num_examples):
    """
    Print the specified number of examples from a JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Print the specified number of examples
    for i, item in enumerate(data[:num_examples], 1):
        print(f"Example {i}:")
        print("Instruction:", item['instruction'])
        print("Input:", item['input'])
        print("Output:", item['output'])
        print()

# Specify the path to the modified JSON file
json_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/alpaca_translate_GPT_35_10_20k.json"

# Specify the number of examples to print
num_examples = 2

# Call the function to print examples from the JSON file
print_examples(json_file_path, num_examples)
