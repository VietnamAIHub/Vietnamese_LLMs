import re
import json 
## combine two files togethers 
all_data = []
with open("vn_0_rdf_result_format_structure.json", 'r') as file:
    data = json.load(file)
with open("vn_1_rdf_result_format_structure.json", 'r') as file:
    data_2 = json.load(file)
for each_data in data:

    all_data.append(each_data)
for each_data in data_2:
    all_data.append(each_data)

with open("Step_by_step_Viet_instructions.json", 'w', encoding='utf-8') as file:
    json.dump(all_data, file, indent=4, ensure_ascii=False)
