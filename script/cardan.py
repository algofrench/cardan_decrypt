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

import numpy as np
import itertools
import unicodedata
import math
import multiprocessing
import sys
import time
import os

# Get the directory of the project
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

grid_size = 4  # Size of the square grids to generate
percentage_threshold = 97
word_size_threshold = 3.6
possible_angles = [[0, 0, 0, 0], [0, 180, 0, 180], [0, 90, 180, 270], [0, 270, 180, 90]]
caesar_num_rotations = 0 # Caesar cipher
dictionary_path = os.path.join(project_dir, 'data', 'words_wiktionary_fr.txt')  # File containing words
exceptions_file = os.path.join(project_dir, 'data', 'words_exceptions.txt')  # File containing exception words
grid_file = os.path.join(project_dir, 'input_grid.txt')  # File containing a specific grid
text_file = os.path.join(project_dir, 'input_text.txt')  # File containing text
max_grid_queue_size = 1000  # Adjust this number based on memory and processing requirements
worker_size = multiprocessing.cpu_count() - 1 # Adjust this number based processing requirements

class TrieNode:
    __slots__ = ('children', 'is_end_of_word')  # Reduce memory usage with __slots__

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0  # Counter to track the number of words in the Trie

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())

        # Only increment the counter if we're marking a new end-of-word
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

    def search(self, word):
        node = self.root
        for char in word:
            node = node.children.get(char)
            if node is None:
                return False
        return node.is_end_of_word

    def remove(self, word):
        """Remove a word from the Trie, decrementing word_count if successful."""
        node = self.root
        stack = []  # Stack to store nodes for backtracking

        # Traverse the Trie to the end of the word
        for char in word:
            if char not in node.children:
                return  # Word not in Trie, no action taken
            stack.append((node, char))
            node = node.children[char]

        # If it's the end of a word, mark it as not an end and update word_count
        if node.is_end_of_word:
            node.is_end_of_word = False
            self.word_count -= 1  # Decrement the word count

        # Clean up nodes that are no longer needed
        while stack:
            parent, char = stack.pop()
            child = parent.children[char]
            if not child.is_end_of_word and not child.children:
                del parent.children[char]  # Delete node if it's not needed
            else:
                break

    def __len__(self):
        return self.word_count

def load_words_into_trie(dictionary_path, exceptions_file=None):
    word_trie = Trie()
    print("Initializing Trie and loading words...")

    # Helper function to add words to the Trie
    def add_words_to_trie(words, description=""):
        count = 0
        for word in words:
            word_trie.insert(word)
            count += 1
        print(f"{description}: {count} words added.")

    # Read the main dictionary file and add each word
    try:
        print(f"Loading dictionary from '{dictionary_path}'...")
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            add_words_to_trie((line.strip() for line in f if line.strip()), description="Dictionary words")
    except FileNotFoundError:
        print(f"Error: Dictionary file '{dictionary_path}' not found.")

    # Add digits directly
    add_words_to_trie((str(i) for i in range(10)), description="Digits")
    
    # Add single-letter words directly
    add_words_to_trie("abcdefghijklmnopqrstuvwxyz", description="Single-letter words")
    
    # Add any specific multi-letter words needed
    add_words_to_trie(["bsm"], description="Specific words")

    # Optionally remove exception words if an exceptions file is provided
    if exceptions_file:
        try:
            print(f"Loading exceptions from '{exceptions_file}'...")
            with open(exceptions_file, 'r', encoding='utf-8') as fe:
                removed_count = 0
                for line in fe:
                    exception_word = line.strip()
                    if exception_word:
                        word_trie.remove(exception_word)
                        removed_count += 1
                print(f"Exceptions: {removed_count} words removed.")
        except FileNotFoundError:
            print(f"Warning: The exceptions file '{exceptions_file}' was not found. No exception words will be removed.")

    print("Trie loading complete.")
    return word_trie

def remove_duplicates(nested_list):
    # Convert inner lists to tuples to make them hashable, then use a set to remove duplicates
    seen = set()
    unique_list = []
    
    for item in nested_list:
        # Convert the inner list (or nested structure) to a tuple
        item_tuple = tuple(map(tuple, item)) if isinstance(item[0], list) else tuple(item)
        
        # Add to result only if it's not already in the set
        if item_tuple not in seen:
            seen.add(item_tuple)
            unique_list.append(item)

    return unique_list

# Function to remove accents and convert to lowercase
def remove_accents_and_lowercase(string):
    # Normalize text in NFD (decomposes combined characters)
    string_no_accents = ''.join(
        char for char in unicodedata.normalize('NFD', string) if unicodedata.category(char) != 'Mn'
    )
    # Convert to lowercase
    return string_no_accents.lower()

def remove_special_characters(string):
    # Remove characters ' . ,
    string = string.replace(" ", "").replace("'", "").replace(".", "").replace(",", "")
    return string

def valid_percentage(string, percentage_threshold=0):
    # Initial cleaning of the text
    string = remove_special_characters(remove_accents_and_lowercase(string.strip()))
    total_letters = len(string)

    if total_letters == 0:
        return string, 0, [], 0

    # Initialize the dynamic programming table with tuples (segmentation, valid_letters)
    dp_table = [([], 0)] * (total_letters + 1)

    max_end, min_invalid_letters = 0, 0
    for start in range(total_letters):
        node = word_trie.root  # Start from the root of the Trie
        for end in range(start, total_letters):
            char = string[end]
            node = node.children.get(char)
            
            if node is None:  # No matching prefix in Trie, break early
                break
            
            if node.is_end_of_word:
                candidate_word = string[start:end + 1]
                max_end = end + 1
                last_segmentation, last_valid_letters = dp_table[start]
                last_word_count = len(last_segmentation)

                # Skip segmentation if it would create three consecutive single-letter words
                if len(candidate_word) == 1 and last_word_count >= 2 and all(len(word) == 1 for word in last_segmentation[-2:]):
                    continue

                # French rules for "qu" that must be followed by vowels
                last_candidate_word = last_segmentation[-1] if last_segmentation else ""
                if last_candidate_word == "qu" and candidate_word[0] not in 'aeiouyh':
                    continue

                # Calculate the total number of valid letters and the new word count
                total_valid_letters = len(candidate_word) + last_valid_letters
                total_words = last_word_count + 1

                # Update the DP table for `end + 1` if we have more valid letters,
                # or the same number of valid letters but fewer words
                current_segmentation, current_valid_letters = dp_table[end + 1]
                current_word_count = len(current_segmentation)

                if total_valid_letters > current_valid_letters or (
                    total_valid_letters == current_valid_letters and total_words < current_word_count
                ):
                    dp_table[end + 1] = (last_segmentation + [candidate_word], total_valid_letters)

        # If no valid segmentation found at this position, consider the character as invalid
        if dp_table[start + 1][1] == 0:
            last_segmentation, last_valid_letters = dp_table[start]
            dp_table[start + 1] = (last_segmentation + ["[" + string[start] + "]"], last_valid_letters)

        # Calculate min_invalid_letters and max_potential_percentage at this point
        if max_end <= start:
            min_invalid_letters += 1
            max_potential_percentage = ((total_letters - min_invalid_letters) / total_letters) * 100
            # Early exit if max_potential_percentage is below threshold
            if max_potential_percentage < percentage_threshold:
                break

    # Retrieve the best segmentation and the percentage of valid letters
    optimal_segmentation, valid_letters = dp_table[total_letters]
    valid_percentage = (valid_letters / total_letters) * 100
    letters_per_word = len(string) / len(optimal_segmentation) if len(optimal_segmentation) > 0 else 0

    return string, letters_per_word, optimal_segmentation, valid_percentage

def display(args):
    (result, letters_per_word, optimal_segmentation, percentage) = args
    print(f"string({len(result):3.0f}) : {result}")
    print(f"Optimal segmentation({percentage:.2f}%, {letters_per_word:.2f}) : {' '.join(optimal_segmentation)}")

def apply_rotation(text, n):
    result = []
    for char in text:
        # Check if the character is an uppercase letter
        if char.isupper():
            # Shift within the uppercase alphabet
            result.append(chr((ord(char) - ord('A') + n) % 26 + ord('A')))
        # Check if the character is a lowercase letter
        elif char.islower():
            # Shift within the lowercase alphabet
            result.append(chr((ord(char) - ord('a') + n) % 26 + ord('a')))
        else:
            # If not a letter, do not modify the character
            result.append(char)
    
    return ''.join(result)

def load_grid(grid_file):
    # Read the grid and convert to numpy matrix
    with open(grid_file, 'r') as f:
        grid = [list(map(int, line.split())) for line in f.readlines()]
    return np.array(grid)

def rotate_grid(grid, angle):
    """ Returns the grid after rotating it `angle` degrees. """
    rotations = angle // 90  # Number of 90-degree rotations
    return np.rot90(grid, k=rotations)

def apply_grid(grid, text, angles, reverse_loops, reset_angle_index):
    grid_rows, grid_cols = grid.shape
    text_rows, text_cols = text.shape
    
    # Create a result grid of the same size as the text, filled with spaces
    result_grid = np.full((text_rows, text_cols), ' ')
    buffer = []

    # Initialize the angle index
    angle_index = 0  

    # Function to process submatrices and apply the grid
    def process_submatrices(i, j, angle_index):
        # Extract the submatrix of text corresponding to the size of the grid
        submatrix = text[i:i + grid_rows, j:j + grid_cols]

        # Apply the corresponding rotation to the grid
        rotated_grid = rotate_grid(grid, angles[angle_index])
        
        # Fill the result grid with the content of the submatrix and the rotated grid
        fill_result_grid(rotated_grid, submatrix, result_grid, i, j, buffer)

    # Traverse the text based on loop options
    if reverse_loops:
        for j in range(0, text_cols, grid_cols):
            if reset_angle_index:
                angle_index = 0
            for i in range(0, text_rows, grid_rows):
                process_submatrices(i, j, angle_index)
                angle_index = (angle_index + 1) % 4
    else:
        for i in range(0, text_rows, grid_rows):
            if reset_angle_index:
                angle_index = 0
            for j in range(0, text_cols, grid_cols):
                process_submatrices(i, j, angle_index)
                angle_index = (angle_index + 1) % 4

    return buffer, result_grid

def fill_result_grid(grid, submatrix, result_grid, i_offset, j_offset, buffer):
    rows, cols = grid.shape
    
    # Apply the grid to the submatrix and put the letters in the result grid
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1 and i < len(submatrix) and j < len(submatrix[i]):
                result_grid[i_offset + i][j_offset + j] = submatrix[i][j]
                buffer.append(submatrix[i][j])

def generate_cardan_grids_on_the_fly(size):
    """ Generator that yields valid square Cardan grids for a given size without storing them in memory. """
    positions = itertools.product([0, 1], repeat=size * size)
    for pos in positions:
        grid = np.array(pos).reshape(size, size)
        
        # Check if any row in the grid has all 1s or has at least three consecutive 1s
        if any(np.all(row == 1) or np.any(np.convolve(row, [1, 1, 1], mode='valid') == 3) for row in grid):
            continue  # Skip this grid if it meets either condition
        
        yield grid  # Yield the grid instead of adding it to a list

def count_valid_rows(n):
    # Generate all possible binary rows of length `n`
    all_rows = list(itertools.product([0, 1], repeat=n))
    
    # Filter rows that don't meet the criteria
    valid_rows = [
        row for row in all_rows 
        if not all(row)  # Exclude rows with all 1s
        and not any(row[i] == row[i+1] == row[i+2] == 1 for i in range(n - 2))  # Exclude rows with 3 consecutive 1s
    ]
    
    return len(valid_rows)

def count_valid_matrices(n):
    valid_rows_count = count_valid_rows(n)
    return valid_rows_count ** n

def display_grid(grid):
    """ Displays a Cardan grid in a readable form """
    for line in grid:
        print(' '.join(map(str, line)))

def convert_to_square(text):
    # Convert text (list of lists) into a single string
    concatenated_text = ''.join([''.join(line) for line in text])

    # Calculate the square root of the total number of characters
    num_chars = len(concatenated_text)
    square_size = math.floor(math.sqrt(num_chars))

    # Fill in missing spaces to get text that can be reshaped into a square
    concatenated_text += ' ' * (square_size**2 - num_chars)

    # Rearrange the text into a square matrix
    square = [concatenated_text[i:i+square_size] for i in range(0, len(concatenated_text), square_size)]

    return [list(line.strip()) for line in square]

def handle_result(displayed_results_set, output_lock, result, grid, angles):
    # Remove accents and convert to lowercase
    result = remove_special_characters(remove_accents_and_lowercase(result))
    if len(result) < 3:
        return
    
    for i in range(caesar_num_rotations + 1):
        string, letters_per_word, optimal_segmentation, percentage = valid_percentage(apply_rotation(result, i), percentage_threshold)
        if percentage < percentage_threshold or letters_per_word < word_size_threshold:
            continue

        with output_lock:
            if optimal_segmentation in displayed_results_set:
                continue
            displayed_results_set.append(optimal_segmentation)

            print()
            display_grid(grid)
            print(f"rot: {i}")
            print(f"angles: {angles}")
            display((string, letters_per_word, optimal_segmentation, percentage))
            print()
            sys.stdout.flush()

def process_grid(displayed_results_set, output_lock, grid, text, possible_angles):
    for reverse_loops in [False, True]:
        for reset_angle_index in [False, True]:
            for angles in possible_angles:
                buffer, result_grid = apply_grid(grid, text, angles, reverse_loops, reset_angle_index)
                
                result = ''.join(buffer)
                handle_result(displayed_results_set, output_lock, result, grid, angles)
                handle_result(displayed_results_set, output_lock, result[::-1], grid, angles)

                buffer = []
                for line in result_grid:
                    buffer.append(''.join(line))
                result = ''.join(buffer)
                handle_result(displayed_results_set, output_lock, result, grid, angles)
                handle_result(displayed_results_set, output_lock, result[::-1], grid, angles)

def worker(displayed_results_set, output_lock, queue, text, possible_angles, total_grids):
    execution_time = 0

    while True:
        start_time = time.time()

        idx, grid = queue.get()
        if idx == -1:
            queue.task_done()
            break

        with output_lock:
            print(f"Cardan Grid {idx + 1}/{total_grids}:", end='\r')
            sys.stdout.flush()
    
        process_grid(displayed_results_set, output_lock, grid, text, possible_angles)
        queue.task_done()

        execution_time += time.time() - start_time
        if execution_time >= 1:
            time.sleep(execution_time * 0.2)
            execution_time = 0

if __name__ == '__main__':

    # Load words into a set
    word_trie = load_words_into_trie(dictionary_path, exceptions_file)
    print(f"Number of unique words: {len(word_trie)}")

    display(valid_percentage("monquatriemesinspiremoncinquiemeestenragemonseptiemedressecrachesonvenin"))
    display(valid_percentage("et l alpha romain. Pour trouver mon tout, il suffit d'être Sage"))
    display(valid_percentage("Mon Premier, première moitié de la moitié du premier âge. Mais, sans protester, suit mon Quatrième et l alpha romain. Pour trouver mon tout, il suffit d'être Sage"))
    display(valid_percentage("Précède mes Second et Troisième, cherchant leur chemin.Mon Sixième, aux limites de l'ETERNITE se cache.Car la Vérité, en vérité, ne sera pas affaire de Devin."))
    display(valid_percentage("quartdelamoitiédehuit"))
    display(valid_percentage("MonPremierpremièremoitiédelamoitiédupremierâge"))
    display(valid_percentage("Ilfaitbeauvictorhugoaimeparisettoulouseetdaboetvillefranchesurmeretmarseille"))
    display(valid_percentage("ILoveParisAndNewYork"))
    display(valid_percentage("iadslkjkjadsbnzcxpoiqwiejkweetalkdfl;kasd"))
    display(valid_percentage("LATLEXT"))
    display(valid_percentage("LATLEXTSJDO"))

    # Load text and convert to 2D matrix
    with open(text_file, 'r') as f:
        text = [list(line.strip()) for line in f.readlines()]

    a = [[x for x in line if x not in [' ']] for line in text]
    b = [[x for x in line if x not in [' ', ',', '.']] for line in text]
    c = [[x for x in line if x not in [' ', ',', '.', "'"]] for line in text]

    texts = [text, a, b, c, convert_to_square(text), convert_to_square(a), convert_to_square(b), convert_to_square(c)]
    texts = remove_duplicates(texts)
    print(f"len(texts): {len(texts)}")
    
    # Calculates the number of n x n matrices with 0/1 entries where no row has all 1s or 3 consecutive 1s
    total_grids = count_valid_matrices(grid_size) + 1
    print(f"total_grids: {total_grids}")

    with multiprocessing.Manager() as manager:
        displayed_results_set = manager.list()  # Create a shared set
        output_lock = multiprocessing.Lock()  # Create a lock object

        # Create a lock object
        output_lock = multiprocessing.Lock()
        
        for text in texts:
            with output_lock:
                print()
                print()
                print("text:")
                print('\n'.join([''.join(line) for line in text]))
                print()
                sys.stdout.flush()

            # Complete short lines with empty spaces to get a rectangular matrix
            max_len = max(len(line) for line in text)
            complete_text = [line + [' '] * (max_len - len(line)) for line in text]
            text_array = np.array(complete_text)

            # Create a work queue
            grid_queue = multiprocessing.JoinableQueue(maxsize=max_grid_queue_size)
            grid_queue.put((1, load_grid(grid_file)))

            # Start worker processes
            processes = []
            for _ in range(worker_size):
                p = multiprocessing.Process(target=worker, args=(displayed_results_set, output_lock, grid_queue, text_array, possible_angles, total_grids))
                processes.append(p)
                p.start()

            # Generate grids and add them to the queue as they are created, respecting the max queue size
            for idx, grid in enumerate(generate_cardan_grids_on_the_fly(grid_size)):
                # with output_lock:
                #     print(f"generated: {idx}")
                #     sys.stdout.flush()
                grid_queue.put((idx + 1, grid))  # `put` will block if the queue is full

            # Send a stop signal for each worker
            for _ in range(worker_size):
                grid_queue.put((-1, None))

            # Wait until all tasks are done
            grid_queue.join()

            # Wait for each process to finish
            for p in processes:
                p.join()
