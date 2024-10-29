# Copyright 2024 algofrench@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xml.sax
import unicodedata
import re
import csv
import os
import bz2
import requests

# Get the directory of the project
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set the language to filter (e.g., "fr" for French)
language = "fr"

# Mapping of language codes to their corresponding "language" terms in Wiktionary
language_terms = {
    "fr": "langue",
    "en": "language",
    # Add more mappings if needed
}

# Get the term for "language" based on the selected language
language_term = language_terms.get(language, "langue")  # Default to "langue" if not found

# Wiktionary download URL for the specified language
download_url = f'https://dumps.wikimedia.org/{language}wiktionary/latest/{language}wiktionary-latest-pages-articles.xml.bz2'

# Path to the compressed XML file (.bz2) in the data directory
xml_file_bz2 = os.path.join(project_dir, 'data', f'{language}wiktionary-latest-pages-articles.xml.bz2')

# Output files
output_file_csv = os.path.join(project_dir, 'data', f'words_wiktionary_{language}.csv')  # CSV file for words and tags
output_file_txt = os.path.join(project_dir, 'data', f'words_wiktionary_{language}.txt')  # TXT file for unique words

# Regular expression to check for Latin characters only
latin_pattern = re.compile(r'^[a-zA-ZÀ-ÿ\'\-]+$')
# Regular expression to capture tags {{...}}
tag_pattern = re.compile(r'=== \{\{S\|(.*?)\}\}')

# Function to remove accents and convert to lowercase
def remove_accents_and_lowercase(string):
    # Normalize the text to NFD (decomposes combined characters)
    no_accents_string = ''.join(
        char for char in unicodedata.normalize('NFD', string) if unicodedata.category(char) != 'Mn'
    )
    # Convert to lowercase
    return no_accents_string.lower()

class WiktionaryHandler(xml.sax.ContentHandler):
    def __init__(self, csv_writer, unique_words, language, language_term):
        self.in_title = False
        self.in_text = False
        self.current_title = []
        self.current_text = []
        self.csv_writer = csv_writer
        self.current_word = None  # Temporarily store the word
        self.unique_words = unique_words  # Store unique words here
        self.language_tag = f"== {{{{{language_term}|{language}}}}}"  # Store the specified language tag

    # Called when an XML element starts
    def startElement(self, tag, attrs):
        if tag == "title":
            self.in_title = True
            self.current_title = []
        elif tag == "text":
            self.in_text = True
            self.current_text = []

    # Called when an XML element ends
    def endElement(self, tag):
        if tag == "title":
            # Store the word for further processing
            self.current_word = "".join(self.current_title).strip()
        elif tag == "text":
            if self.current_word and latin_pattern.match(self.current_word):
                if self.current_word != self.current_word.upper():
                    word = remove_accents_and_lowercase(self.current_word)
                    
                    # Check if the section contains the specified language content
                    full_text = "".join(self.current_text)
                    if self.language_tag in full_text:
                        # Extract the tags {{...}} to add as columns
                        tags = set(tag_pattern.findall(full_text))
                        tags.discard("voir aussi")

                        # Write the row to the CSV with the word and extracted tags
                        self.csv_writer.writerow([word] + list(tags))

                        # Add the word to the set for unique words
                        self.unique_words.add(word)
            # Reset after processing the text
            self.in_text = False
            self.current_word = None
        self.in_title = False

    # Called for each text segment found in the element
    def characters(self, content):
        if self.in_title:
            self.current_title.append(content)
        if self.in_text:
            self.current_text.append(content)

# Check if the .bz2 file is already downloaded; if not, download it
if not os.path.exists(xml_file_bz2):
    print(f"{xml_file_bz2} not found. Downloading from {download_url}...")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses

    # Save the .bz2 file to the data directory
    with open(xml_file_bz2, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Download complete.")

# Create a SAX parser
parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)

# Set to store unique words
unique_words = set()

# Open the output CSV file once and initialize the CSV writer
with open(output_file_csv, 'w', encoding='utf-8', newline='') as f_csv:
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(["word", "tags"])  # Write column headers

    # Apply the custom handler to process the file
    handler = WiktionaryHandler(csv_writer, unique_words, language, language_term)
    parser.setContentHandler(handler)

    # Start parsing the compressed XML file (.bz2)
    print("Extracting words from the compressed XML file...")
    with bz2.open(xml_file_bz2, 'rt', encoding='utf-8') as f_bz2:
        parser.parse(f_bz2)
    print("Extraction complete.")

# Write the unique words to the text file
with open(output_file_txt, 'w', encoding='utf-8') as f_txt:
    for word in sorted(unique_words):  # Sort words alphabetically
        f_txt.write(f"{word}\n")

print("Unique words written to TXT file.")
