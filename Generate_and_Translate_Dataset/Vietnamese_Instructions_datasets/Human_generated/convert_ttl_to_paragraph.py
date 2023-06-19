from rdflib import Graph
import json


def remove_prefixes(turtle_file, output_file):
    with open(turtle_file, 'r') as file:
        turtle_data = file.read()

    # Remove the lines containing prefix declarations
    lines = turtle_data.split('\n')
    cleaned_lines = [line for line in lines if not line.startswith('@prefix')]
    
    cleaned_turtle = '\n'.join(cleaned_lines)
    
    with open(output_file, 'w') as file:
        file.write(cleaned_turtle)
#remove_prefixes('vn_0_rdf_result.ttl', 'vn_0_rdf_result_cleaned.ttl')
# breakpoint()



import re
import json

# # Read the .ttl file
# with open("vn_0_rdf_result.ttl", 'r') as file:
#     #turtle_data = file.read()
#     turtle_data = file.readlines()
# breakpoint()

# topic_matches = []
# for i, line in enumerate(turtle_data):
#     match = re.search(r'(w:[^ ]+)\s+rdf:type prohow:instruction_set', line)
#     if match:
#         topic_matches.append((match.group(1), i))

# #topic_matches = re.findall(r'(w:[^ ]+)\s+rdf:type prohow:instruction_set', turtle_data)
# breakpoint()
# matches = re.findall(r'(w:[^ ]+)\s+(?:rdfs:label|dbo:abstract)\s+"""([^"]+)"""@vn', turtle_data)

# topics = []
# current_topic = []

# for match in matches:
#     prefix = match[0]
#     content = match[1]
#     current_topic.append((prefix, content))
    
#     if 'rdf:type prohow:instruction_set' in content:
#         if current_topic:
#             topics.append(current_topic)
#             current_topic = []

# # Append the remaining topic if there is any
# if current_topic:
#     topics.append(current_topic)

# # Create a list to store the final data
# data = []

# # Process each topic and generate the JSON data
# for topic in topics:
#     topic_data = {
     
#         "contents": []
#     }

#     for item in topic:
#         prefix = item[0]
#         content = item[1]
    
#         topic_data["contents"].append(content)

#     data.append(topic_data)

# # Save the data to a JSON file
# with open("output_2.json", 'w', encoding='utf-8') as file:
#     json.dump(data, file, ensure_ascii=False,indent=4)

from bs4 import BeautifulSoup
# Read the .ttl file
with open("vn_0_rdf_result.ttl", 'r') as file:
    turtle_data = file.readlines()
    #turtle_data = file.read()

topics = []
current_topic = None

for i, line in enumerate(turtle_data):
    index=i
    if 'rdf:type prohow:instruction_set' in line:
        if current_topic is not None:
            topics.append(current_topic)
        current_topic = []

    elif current_topic is not None:
        #match = re.search(r'"""([^"]+)"""@vn', line)
        if line.startswith('<sup'):
            line = re.sub(r'<sup[^>]*>[^<]+</sup>', '', line)
        
        # elif line.startswith('<li>'):
        #     print(line)
        #     break
        else:
            #print("String does not start with <sup")
            line=line

        match = re.search(r'(?:rdfs:label|dbo:abstract|<i>|</i>|<li>|<ul>)([^<]+)(?:</ul>)?|<sup[^>]*>([^<]+)</sup>|"""([^"]+)"""@vn', line)

        #match = re.search(r'(?:rdfs:label|dbo:abstract|<i>|</i>|<ul>)([^<]+)(?:</ul>)?|<sup[^>]*>([^<]+)</sup>|"""([^"]+)"""@vn', line)
        #match = re.search(r'(?:rdfs:label|dbo:abstract|<ul>)([^<]+)(?:</ul>)?|<sup[^>]*>([^<]+)</sup>|"""([^"]+)"""@vn', line)
        # breakpoint()
        if match:
            
            content = match.group(1) or match.group(2) or match.group(3)

            #soup = BeautifulSoup(content, 'html.parser')
            if 'rdfs:label' in line:
                content = f"Tiêu đề: {content}"
            elif 'dbo:abstract' in line:
                content = f"Tóm tắt : {content}"
        
            soup = BeautifulSoup(content, 'html.parser')
            cleaned_content = soup.get_text().strip('\""')
            cleaned_content=cleaned_content.replace('\n','')
            cleaned_content = cleaned_content.replace('@vn','')

            cleaned_content = re.sub(r'<[^>]+>', '', cleaned_content)
            # Check if the row starts with "w:"
            if not line.startswith("w:"):
                # print("condition match")
                cleaned_content = line.strip()
            # if '<sup' in content:
            #     sup_tags = soup.find_all('sup')
            #     first_sup_tag_text = sup_tags[0].get_text()
            #     cleaned_content = cleaned_content.replace(first_sup_tag_text, '')
           
          
            # Check if the content contains <sup> tags
          
            # Remove HTML tags in the content
           
            content = cleaned_content.strip("\"\"\"")
            content = content.strip("@vn ")
            current_topic.append(content)


# Append the last topic if there is any
if current_topic is not None:
    topics.append(current_topic)

# Save the data to a JSON file
with open("vn_0_rdf_result_format.json", 'w', encoding='utf-8') as file:
    json.dump(topics, file, ensure_ascii=False, indent=4)


