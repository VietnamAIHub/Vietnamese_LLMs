"""
@@TranNhiem 
Update the Human instruction step by Step dataset for Wikihow dataset 
This Code feature 
+ 1 Removing the prefix of the HTML tag, "English Text:  \"\"\"Main Steps\"\"\" .",
+ 2 Adding the Structutre & Response Step by Step 
+ 3 Adding the Title of the Instruction


"""
##*******************************************************************************************
### 1 Processing Wikihow dataset 
##*******************************************************************************************

import json
import string 
import re
from bs4 import BeautifulSoup
# Read the JSON file
with open('vn_1_rdf_result_format.json', 'r') as file:
    data = json.load(file)


# Create an empty list to store the dictionaries
dictionary_list = []
# Iterate over the nested lists

topic_list=[]
for sublist in data:
    # Iterate over the items in each sublist
     # Check if the sublist is not empty
    if sublist:
        # Get the first item from the sublist
        first_line = sublist[0].replace('Tiêu đề: ', '')
        first_line = first_line.replace('"""', '').strip('"')

        second_line = sublist[1].replace('Tóm tắt : ', '')
        second_line = second_line.replace('"""', '').strip('"')

        count = 0
        response =[]
        summary_set=set()
        print("prompt: ", first_line)
        topic_list.append(first_line)
        

        # Initialize variables
        #title = None
        #summary = None
        details = []
        detailed_set=set()
        new_list=[]
        
        for i, each_line in enumerate(sublist[2:]):
            if each_line != 'Tiêu đề:  """Main Steps""" .':
                new_list.append(each_line)
        
        #breakpoint()
        current_title_pair=[]
        current_abstract_pair=[]
        for i, each_line in enumerate(new_list):
            print(each_line)

            if each_line.startswith('Tiêu đề:'):
                
                title = each_line.replace('Tiêu đề:', "")
                title = title.replace('"""', '').strip('"')
                title = title.replace(' .', '').strip()
                title= re.sub(r'<sup.*?</sup>', '', title)
                title= re.sub(r'.<sup.*?</sup>', '.', title)

                current_title_pair.append(title)

            elif each_line.startswith('Tóm tắt :'):
                summary = each_line.replace('Tóm tắt :', '').strip('"')
                summary=summary.replace('"""', '').strip('"')
                summary= re.sub(r'<sup.*?</sup>', '', summary)
                summary= re.sub(r'.<sup.*?</sup>', '.', summary)

                if summary.isspace() or not summary: 
                    summary="-"
                current_abstract_pair.append(summary)

            if each_line.startswith('<li>'):
                detail=each_line.replace('<li>', '-').strip('')
                detail=detail.replace('</li>', '').strip('"')
                detailed= re.sub(r'<sup.*?</sup>', '', detail)
                detailed= re.sub(r'.<sup.*?</sup>', '.', detail)

                detail=detail.replace('"""', '').strip('"')

                details.append(detail)

            if each_line.startswith('Tóm tắt :') and len(current_abstract_pair)==2:
                count += 1
                
                #print(current_abstract_pair)
                #print("this is title list", current_title_pair)
                len_list=len(current_title_pair)

                title_ = current_title_pair[0]
                title  =current_title_pair[0]
                title=title.replace('.', '')
                title= str(count)+ "." + title
                #print(title)
                summary = current_abstract_pair[0]
                #breakpoint()
                # if title: 
                # Concatenate title and summary
                if title and len(details)>0:
                    print("condition is match")
                    detail = "\n".join(details)
                    # if detail not in detailed_set:
                    #     detailed_set.add(detail)
                    
                    #     title_summary = title + ':' + summary + '(' + detail + ')'
                    # else:
                    #     continue
                    #title=None
                    # summary=None
                    if summary == "-":
                        title_summary_ = title_ + ':' +  '(' + detail + ')'
                        title_summary = title + ':' + '(' + detail + ')'
                    else:
                        title_summary_ = title_ + ':' + summary +'(' + detail + ')'
                        title_summary = title + ':' + summary + '(' + detail + ')'
                    details=[]
                    title=None
                
                elif title and summary == "-" and len(details) == 0:
                #     # if summary is not "":
                #     # title_summary = title + ':' + summary
                    continue
                else: 
                    title_summary_ = title_ + ':' + summary 
                    title_summary = title + ':' + summary 
                        # summary=None
                
                    # Check if the summary already exists in the set
                    #details ="\n".join(details)
                    # if summary not in summary_set :#and details not in detailed_set
                    #     #response.add(second_line)
                if title_summary_ not in summary_set:
                    #breakpoint()
                    summary_set.add(title_summary_)
                    title_summary= re.sub(r'<sup.*?</sup>', '', title_summary)
                    title_summary = BeautifulSoup(title_summary, "html.parser")
                    title_summary = title_summary.get_text()
                    response.append(title_summary)
                    # summary_set.add(summary)
                    #breakpoint()
                current_abstract_pair=[current_abstract_pair[1]]
                # if len(current_title_pair) > 2:
                #     current_title_pair=[current_title_pair[-1]]
                # else:
                current_title_pair=[current_title_pair[-1]]
                    
                        #detailed_set.add(details)
                    #response.append(title_summary)
                    #response.add(title_summary)
                    # title = None
                    # summary = None
                    #details = []
                #breakpoint()
            if (i == len(sublist[2:]) - 1):
                count += 1
                 # and len(current_abstract_pair)==1:
                print("condition is matched")
                print(current_abstract_pair)
                summary = current_abstract_pair[-1]
                title = current_title_pair[-1]
                title= str(count)+ "." + title
                print("This is last line", title)
                if title: 
                    # Concatenate title and summary
                    if title and details:
                        detail = "\n".join(details)
                        # if detail not in detailed_set:
                        #     detailed_set.add(detail)
                        
                        #     title_summary = title + ':' + summary + '(' + detail + ')'
                        # else:
                        #     continue
                        #title=None
                        # summary=None
                        title_summary = title + ':' + summary + '(' + detail + ')'
                        details=[]
                    else:
                        title_summary = title + ':' + summary 
                        # summary=None
                    
                    # if title_summary not in summary_set:
                    #     summary_set.add(title_summary)
                    #     response.append(title_summary)
                    
                    # Check if the summary already exists in the set
                    #details ="\n".join(details)
                    # if summary not in summary_set :#and details not in detailed_set
                    #     #response.add(second_line)
            
       
        # breakpoint()
 
        # Create a dictionary with "prompt" and "response" keys
        # response=list(response)
        response.insert(0,second_line)
        dictionary = {"prompt": first_line, "response": response}
        dictionary_list.append(dictionary)
        # breakpoint()
output_path="./vn_1_rdf_result_format_structure.json"
with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(dictionary_list, outfile, ensure_ascii=False, indent=4)
    print(f"\nData saved to {output_path} with readable Vietnamese format.")
output_path_topic="./vn_1_rdf_result_format_topic.json"
with open(output_path_topic, 'w', encoding='utf-8') as outfile:
    json.dump(topic_list, outfile, ensure_ascii=False, indent=4)
    print(f"\nData saved to {output_path} with readable Vietnamese format.")

##*******************************************************************************************
### 2 ViQuaAD dataset
##*******************************************************************************************

import json

# def explore_json_structure(file_path):
#     """
#     Reads and explores the structure of a JSON file.

#     Args:
#         file_path (str): Path to the JSON file.
#     """

#     with open(file_path, 'r') as file:
#         data = json.load(file)

#     # Save the data to a new JSON file
#     output_path = "/home/rick/Vietnamese_LLMs/Vietnamese_LLMs/Generate_and_Translate_Dataset/Vietnamese_Instructions_datasets/Human_generated/ViQuAD1.0/test_ViQuAD_format.json"
#     with open(output_path, 'w', encoding='utf-8') as outfile:
#         json.dump(data, outfile, ensure_ascii=False, indent=4)
#         print(f"\nData saved to {output_path} with readable Vietnamese format.")
#     # Iterate over each data object
#     for item in data.get('data', []):
#         # Print the title
#         title = item.get('title')
#         print(f"\nTitle: {title}")
#         breakpoint()
#         # Iterate over each paragraph
#         paragraphs = item.get('paragraphs', [])
#         for paragraph in paragraphs:
#             # Iterate over each question-answer pair
#             qas = paragraph.get('qas', [])
#             for qa in qas:
#                 # Print the question
#                 question = qa.get('question')
#                 print(f"\nQuestion: {question}")

#                 # Iterate over each answer
#                 answers = qa.get('answers', [])
#                 for answer in answers:
#                     # Print the answer
#                     answer_text = answer.get('text')
#                     answer_start = answer.get('answer_start')
#                     print(f"Answer: {answer_text}")
#                     print(f"Answer Start: {answer_start}")
# explore_json_structure("/home/rick/Vietnamese_LLMs/Vietnamese_LLMs/Generate_and_Translate_Dataset/Vietnamese_Instructions_datasets/Human_generated/ViQuAD1.0/test_ViQuAD.json")
