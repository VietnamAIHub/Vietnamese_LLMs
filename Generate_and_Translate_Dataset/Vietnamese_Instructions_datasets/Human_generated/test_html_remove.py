# from bs4 import BeautifulSoup

# html = '''
#     <sup id="_ref-13" class="reference" aria-label="Liên kết tới Nguồn 13"><a href="#_note-13">[13]</a></sup>
#     <ul>
#         <li>Nếu bạn hay dị ứng, đang có vấn đề về sức khỏe, đang uống thuốc khác, có thai hay nuôi con bằng sữa mẹ thì phải luôn cho bác sĩ biết trước khi bắt đầu bất kì quá trình điều trị nào.</li>
#         <li>Không được sử dụng dược phẩm cho trẻ em mà không tham khảo ý kiến bác sĩ khoa nhi trước. Điều trị chàm da đầu ở trẻ em là một quá trình khác và sẽ được đề cập trong bài viết này.</li>
#     </ul>
# '''

# soup = BeautifulSoup(html, 'html.parser')

# text = soup.get_text()
# print(text)

# import re

# # Read the .ttl file
# with open("vn_0_rdf_result.ttl", 'r') as file:
#     turtle_data = file.read()

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

# # Print the separated topics
# for topic in topics:
#     print("Topic:")
#     for item in topic:
#         prefix = item[0]
#         content = item[1]
#         print(f"Prefix: {prefix}")
#         print(f"Content: {content}")
#         print('---')
#     print('=====================')
#     breakpoint()




# import re
# import json
# import sys

# # Set the encoding of standard output to UTF-8
# sys.stdout.reconfigure(encoding='utf-8')

# # Read the .ttl file
# with open("vn_0_rdf_result.ttl", 'r', encoding='utf-8') as file:
#     turtle_data = file.read()

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
#         "instruction": "",
#         "response": []
#     }

#     for item in topic:
#         prefix = item[0]
#         content = item[1]
#         if 'rdfs:label' in prefix:
#             topic_data["instruction"] = content
#         else:
#             topic_data["response"].append(content)

#     data.append(topic_data)

# # Save the data to a JSON file
# with open("output.json", 'w', encoding='utf-8') as file:
#     json.dump(data, file, indent=4, ensure_ascii=False)

# import re
# import json
# # Read the .ttl file
# with open("vn_0_rdf_result.ttl", 'r') as file:
#     turtle_data = file.read()

# matches = re.findall(r'(w:[^ ]+)\s+(?:rdfs:label|dbo:abstract)\s+"""([^"]+)"""@vn', turtle_data)

# topics = []
# current_topic = None

# for match in matches:
#     prefix = match[0]
#     content = match[1]
    
#     if 'rdf:type prohow:instruction_set' in content:
       
#         if current_topic is not None:
#             topics.append(current_topic)
#         current_topic = []
    
#     if current_topic is not None:
#         current_topic.append(content)

# # Append the last topic if there is any
# if current_topic is not None:
#     topics.append(current_topic)

# # Save the data to a JSON file
# output = {"topics": []}
# for topic in topics:
#     output["topics"].append({"instruction": topic[0], "response": topic[1:]})

# with open("output_2.json", 'w', encoding='utf-8') as file:
#     json.dump(output, file, ensure_ascii=False, indent=4)


import re
import json

# Read the .ttl file
with open("vn_0_rdf_result.ttl", 'r') as file:
    turtle_data = file.read()

topic_matches = re.findall(r'(w:[^ ]+)\s+rdf:type prohow:instruction_set', turtle_data)

topics = []
for match in topic_matches:
    prefix = match.strip()
    content_match = re.search(rf'{prefix}\s+(?:rdfs:label|dbo:abstract)\s+"""([^"]+)"""@vn', turtle_data)
    if content_match:
        content = content_match.group(1)
        topics.append(content)

# Save the data to a JSON file
with open("output.json", 'w') as file:
    json.dump(topics, file, ensure_ascii=False, indent=4)
