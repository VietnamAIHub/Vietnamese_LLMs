'''
TranNhiem 2023/06/10
Preprocessing text before translating it using multiple the preprocess_text function.

'''



import spacy
import re 
import string
import json
import pandas as pd

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

def remove_urls(text):
    """
    Remove URLs from the text using SpaCy.

    Args:
        text (str): Input text.

    Returns:
        str: Text with URLs removed.
    """
    doc = nlp(text)
    text_without_urls = " ".join([token.text for token in doc if not token.like_url])
    return text_without_urls

def remove_html_tags(text):
    """
    Remove HTML tags from the text using regular expressions.

    Args:
        text (str): Input text.

    Returns:
        str: Text with HTML tags removed.
    """
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def matches_regex(regex, text):
    """
    Check if the text matches the given regex pattern.

    Args:
        regex (str): Regular expression pattern.
        text (str): Input text.

    Returns:
        bool: True if the text matches the pattern, False otherwise.
    """
    return bool(re.compile(regex).search(text))

def remove_special_characters(text, keep_chars="'.,!?"):
    """
    Remove special characters from the text, except for the specified keep_chars.

    Args:
        text (str): Input text.
        keep_chars (str): Characters to keep in the text.

    Returns:
        str: Text with special characters removed.
    """
    pattern = re.compile(f'[^A-Za-z0-9{keep_chars}\s]')
    return pattern.sub(r'', text)

def contains_code(text):
    """
    Check if the text contains code snippets.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text contains code, False otherwise.
    """
    code_blacklist = ['&&', '||', '<html>', ';\n', 'SELECT']
    return (
        any(code_keyword in text for code_keyword in code_blacklist) or
        matches_regex(r'\w+\(\w*\) \{', text) or
        matches_regex(r'def \w+\(', text) or
        matches_regex(r'\[A-z]+\.[A-z]+', text) or
        matches_regex(r': [\w\.#]{1,12};', text) or
        matches_regex(r'<\/\w+>', text)
    )

def preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False,
                    stemming=False, lemmatization=False, keep_chars="'.,!?", remove_code=False):
    """
    Preprocess the text by removing URLs, HTML tags, special characters, digits, punctuation,
    and applying lowercase, stopword removal, stemming, or lemmatization if specified.

    Args:
        text (str): Input text.
        remove_digits (bool): Whether to remove digits from the text.
        to_lowercase (bool): Whether to convert the text to lowercase.
        remove_stopwords (bool): Whether to remove stopwords using SpaCy.
        stemming (bool): Whether to perform stemming using SpaCy's Lemmatizer.
        lemmatization (bool): Whether to perform lemmatization using SpaCy's Lemmatizer.
        keep_chars (str): Characters to keep in the text.
        remove_code (bool): Whether to remove code snippets from the text.

    Returns:
    str: Processed text.
    """
    def remove_punctuation(text):
        return ''.join(c if c not in string.punctuation or c == '-' else ' ' for c in text)

    # Remove URLs using SpaCy
    text = remove_urls(text)

    # Remove HTML tags
    text = remove_html_tags(text)

    # Remove special characters
    text = remove_special_characters(text, keep_chars=keep_chars)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove code content
    if remove_code:
        text = re.sub(r'(?s)(?P<tag><code>.*?</code>)', '', text)

    if remove_digits:
        text = re.sub(r'\d+', '', text)

    if to_lowercase:
        text = text.lower()

    # Call the remove_punctuation function
    text = remove_punctuation(text)

    if remove_stopwords or stemming or lemmatization:
        # Tokenize the text using SpaCy
        doc = nlp(text)

        if remove_stopwords:
            # Remove stop words using SpaCy
            tokens = [token.text for token in doc if not token.is_stop]
        else:
            tokens = [token.text for token in doc]

        if stemming:
            # Perform stemming using SpaCy's Lemmatizer
            tokens = [token.lemma_ for token in doc]

        if lemmatization:
            # Perform lemmatization using SpaCy's Lemmatizer
            tokens = [token.lemma_ for token in doc]

        text = ' '.join(tokens)

    return text

# Check if the given text contains words
def contains_words(text):
    return matches_regex(r'[A-z]{3,}', text)

# Check if the given text is translatable
def is_translatable(text):
    if text == "":
        return False
    return (contains_code(text) is False) and contains_words(text)

def contains_words(text):
    """
        Check if the text contains words.

        Args:
            text (str): Input text.

        Returns:
            bool: True if the text contains words, False otherwise.
    """
    return matches_regex(r'[A-z]{3,}', text)

def is_translatable(text):
    """
    Check if the given text is translatable.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text is translatable, False otherwise.
    """
    if text == "":
        return False
    return (contains_code(text) is False) and contains_words(text)


# Load input data as DataFrame
def load_input_data(INPUT_TASKS_PATH):
    """
    Load input data from a JSON file and return as a DataFrame.

    Args:
        INPUT_TASKS_PATH (str): Path to the input JSON file.

    Returns:
        pd.DataFrame: Input data as a DataFrame.
    """
    with open(INPUT_TASKS_PATH, "rb") as f:
        json_data = json.loads(f.read())
    return pd.DataFrame(json_data)


## Dealing Json dataset with Index with Convert
def load_input_data_index(INPUT_TASKS_PATH):
    with open(INPUT_TASKS_PATH, 'r') as json_file:
        content = json_file.read()

    # Split the content by line and remove empty lines
    json_objects = [line for line in content.splitlines() if line.strip()]

    df_list = []  # List to store DataFrames for each JSON object

    # Iterate through the JSON objects, load and convert them into DataFrames
    for index, json_object in enumerate(json_objects):
        try:
            data = json.loads(json_object)
            df = pd.DataFrame([data], index=[index])  # Convert JSON object to DataFrame with index
            df_list.append(df)  # Append DataFrame to list
        except (json.JSONDecodeError, ValueError) as err:
            print(f"Error parsing JSON Object {index + 1}: {err}")

    # Concatenate the DataFrames in the list into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)
    print(f"Complete Loaded {len(final_df)} JSON objects.")
    return final_df
