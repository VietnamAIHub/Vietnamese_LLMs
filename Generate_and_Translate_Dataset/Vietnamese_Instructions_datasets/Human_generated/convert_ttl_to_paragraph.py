from bs4 import BeautifulSoup
import json
import re 
##*******************************************************************************************
### 1 Processing Wikihow dataset 
##*******************************************************************************************


def format_ttl_file(input_file, output_file):
    """
    Formats a .ttl file containing Turtle data and saves the formatted data to a JSON file.

    Args:
        input_file (str): Path to the input .ttl file.
        output_file (str): Path to the output JSON file.
    """

    # Read the .ttl file
    with open(input_file, 'r') as file:
        turtle_data = file.readlines()

    topics = []  # List to store the formatted topics
    current_topic = None  # Current topic being processed

    # Iterate over each line in the .ttl file
    for i, line in enumerate(turtle_data):
        if 'rdf:type prohow:instruction_set' in line:
            # Start of a new topic
            if current_topic is not None:
                topics.append(current_topic)  # Add the previous topic to the list
            current_topic = []  # Start a new topic

        elif current_topic is not None:
            # Process the lines within a topic

            # Remove <sup> tags from the line
            if line.startswith('<sup'):
                line = re.sub(r'<sup[^>]*>[^<]+</sup>', '', line)

            # Extract content using regular expressions
            match = re.search(r'(?:rdfs:label|dbo:abstract|<i>|</i>|<li>|<ul>)([^<]+)(?:</ul>)?|<sup[^>]*>([^<]+)</sup>|"""([^"]+)"""@vn', line)
            if match:
                content = match.group(1) or match.group(2) or match.group(3)

                # Identify and format content based on the line type
                if 'rdfs:label' in line:
                    content = f"Tiêu đề: {content}"
                elif 'dbo:abstract' in line:
                    content = f"Tóm tắt : {content}"

                # Clean the HTML tags and special characters using BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                cleaned_content = soup.get_text().strip('\""')
                cleaned_content = cleaned_content.replace('\n', '')
                cleaned_content = cleaned_content.replace('@vn', '')
                cleaned_content = re.sub(r'<[^>]+>', '', cleaned_content)

                # Check if the row starts with "w:"
                if not line.startswith("w:"):
                    cleaned_content = line.strip()

                current_topic.append(cleaned_content)  # Add the cleaned content to the current topic

    # Append the last topic if there is any
    if current_topic is not None:
        topics.append(current_topic)

    # Save the data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(topics, file, ensure_ascii=False, indent=4)

# format_ttl_file("vn_0_rdf_result.ttl", "vn_0_rdf_result_format.json")


##*******************************************************************************************
### 2 ViQuaAD dataset
##*******************************************************************************************

