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

grid_size = 2  # Size of the square grids between 2 and 5 (don't try with more than 5, you need 16GB of memory to generate and store the 28 millions 5x5 cardan grids)
percentage_threshold = 97
word_size_threshold = 3.6
possible_angles = [[0, 0, 0, 0], [0, 180, 0, 180], [0, 90, 180, 270], [0, 270, 180, 90]]
caesar_num_rotations = 0 # Caesar cipher
dictionary_path = os.path.join(project_dir, 'data', 'words_wiktionary_fr.txt')  # File containing words
exceptions_file = os.path.join(project_dir, 'data', 'words_exceptions.txt')  # File containing exception words
grid_file = os.path.join(project_dir, 'input_grid.txt')  # File containing a specific grid
text_file = os.path.join(project_dir, 'input_text.txt')  # File containing text

def save_binary_grids(grids, binary_grid_file, grid_size):
    # Calculate the number of bits needed (grid_size^2 bits per grid)
    bits_per_grid = grid_size ** 2

    # Prepare binary content
    binary_content = bytearray()

    for grid in grids:
        bits = 0
        for i in range(grid_size):
            for j in range(grid_size):
                # Shift bits to the left and add 1 or 0 based on grid position
                bits = (bits << 1) | int(grid[i, j])  # Convert to native int to avoid errors

        # Convert to bytes and add to binary content
        binary_content.extend(bits.to_bytes((bits_per_grid + 7) // 8, byteorder='big'))

    # Write the encoded grids to the binary file
    with open(binary_grid_file, 'wb') as f:
        f.write(binary_content)

def load_binary_grids(binary_grid_file, grid_size):
    # Calculate the number of bits needed (grid_size^2 bits per grid)
    bits_per_grid = grid_size ** 2

    grids = []
    with open(binary_grid_file, 'rb') as f:
        binary_content = f.read()
        # Decode each grid from binary
        for i in range(0, len(binary_content), (bits_per_grid + 7) // 8):
            bits = int.from_bytes(binary_content[i:i + (bits_per_grid + 7) // 8], byteorder='big')
            grid = np.zeros((grid_size, grid_size), dtype=int)
            for j in range(bits_per_grid):
                # Extract each bit to fill the grid
                grid[grid_size - 1 - (j // grid_size), grid_size - 1 - (j % grid_size)] = (bits >> j) & 1
            grids.append(grid)
    return grids

def generate_or_load_grids(grid_size):
    binary_grid_file = os.path.join(project_dir, 'data', f'grids_cardan_{grid_size}x{grid_size}.bin')
    try:
        # Try to load the binary file if it exists
        grids = load_binary_grids(binary_grid_file, grid_size)
    except FileNotFoundError:
        # Generate grids and save them if the file doesn't exist
        grids = generate_cardan_grids(grid_size)
        save_binary_grids(grids, binary_grid_file, grid_size)
    return grids

def load_words_into_set(dictionary_path):
    words = set()  # Create an empty set
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()  # Remove leading and trailing spaces
            if word:  # Ignore empty lines
                words.add(word)  # Add the word to the set
    
    # Add digits
    words.update(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    
    # Add desired single-letter words
    words.update(["a", "c", "j", "l", "m", "n", "s", "t", "y"])
    
    # Add desired multi-letter words
    words.add("bsm")

    # Read exception words from the file
    try:
        with open(exceptions_file, 'r', encoding='utf-8') as fe:
            for line in fe:
                exception_word = line.strip()
                if exception_word in words:  # Remove the word only if it's in the set
                    words.remove(exception_word)
    except FileNotFoundError:
        print(f"The file '{exceptions_file}' was not found. No exception words will be removed.")

    return words

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

def valid_percentage(string):
    # Initial cleaning of the text
    string = remove_special_characters(remove_accents_and_lowercase(string.strip()))
    total_letters = len(string)

    # DP table to store the best segmentations and valid letters for each position
    dp_table = [None] * (total_letters + 1)
    dp_table[0] = ([], 0)  # Initialization: 0 words and 0 valid letters at the start

    for j in range(1, total_letters + 1):
        best_segmentation = None
        max_valid_letters = 0

        # Iterate over each starting index `i` for the substring ending at `j`
        for i in range(j):
            candidate_word = string[i:j]

            # Check if the candidate word is in the set of valid words
            if candidate_word in word_set:
                last_segmentation, last_valid_letters = dp_table[i]
                last_candidate_word = last_segmentation[-1] if last_segmentation else ""

                # Specific french rules for single-letter words
                if len(candidate_word) == 1 and candidate_word not in["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    if candidate_word not in ["a", "y", "l", "s", "j", "m", "n", "t", "c"] or j == total_letters:
                        continue
                    if len(last_candidate_word) == 1:
                        if last_candidate_word in ["l", "m", "n", "t"] and candidate_word not in ['a', 'y']:
                            continue
                        if last_candidate_word in ["s", "j"] and candidate_word != 'y':
                            continue
                        if last_candidate_word == candidate_word:
                            continue
                # French rules for letters that must be followed by vowels
                if last_candidate_word in ["l", "s", "j", "m", "n", "t", "qu"] and candidate_word[0] not in 'aeiouyh':
                    continue
                if last_candidate_word == "c" and candidate_word[0] not in 'aeiouy':
                    continue

                # Calculate the total number of valid letters and update the optimal segmentation
                total_valid_letters = len(candidate_word) + last_valid_letters
                if total_valid_letters > max_valid_letters or (total_valid_letters == max_valid_letters and (best_segmentation is None or len(best_segmentation) > len(last_segmentation) + 1)):
                    best_segmentation = last_segmentation + [candidate_word]
                    max_valid_letters = total_valid_letters

        # If no valid segmentation found, consider the character as invalid
        if best_segmentation is None:
            last_segmentation, last_valid_letters = dp_table[j - 1]
            best_segmentation = last_segmentation + ["[" + string[j - 1] + "]"]
            max_valid_letters = last_valid_letters

        dp_table[j] = (best_segmentation, max_valid_letters)

    # Retrieve the best segmentation and the percentage of valid letters
    optimal_segmentation, valid_letters = dp_table[total_letters]
    valid_percentage = (valid_letters / total_letters) * 100 if total_letters > 0 else 0
    letters_per_word = len(string) / len(optimal_segmentation)

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

def generate_cardan_grids(size):
    """ Generates all possible square Cardan grids for a given size. """
    positions = list(itertools.product([0, 1], repeat=size * size))
    grids = []
    for pos in positions:
        grid = np.array(pos).reshape(size, size)
        
        # Check if any row in the grid has all 1s
        if any(np.all(row == 1) for row in grid):
            continue  # Skip this grid if it has a full row of 1s
        
        grids.append(grid)
    return grids

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
        string, letters_per_word, optimal_segmentation, percentage = valid_percentage(apply_rotation(result, i))
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
    word_set = load_words_into_set(dictionary_path)
    print(f"Number of unique words: {len(word_set)}")

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

    # Load or generate grids
    cardan_grids = [load_grid(grid_file)]
    cardan_grids.extend(generate_or_load_grids(grid_size))
    total_grids = len(cardan_grids)

    # Load text and convert to 2D matrix
    with open(text_file, 'r') as f:
        text = [list(line.strip()) for line in f.readlines()]

    a = [[x for x in line if x not in [' ']] for line in text]
    b = [[x for x in line if x not in [' ', ',', '.']] for line in text]
    c = [[x for x in line if x not in [' ', ',', '.', "'"]] for line in text]

    texts = [text, a, b, c, convert_to_square(text), convert_to_square(a), convert_to_square(b), convert_to_square(c)]
    texts = remove_duplicates(texts)
    print(f"len(texts): {len(texts)}")

    with multiprocessing.Manager() as manager:
        displayed_results_set = manager.list()  # Create a shared set

        # Create a lock object
        output_lock = multiprocessing.Lock()
        
        # Start a process for each text
        processes = []
        for text in texts:
            print()
            print()
            print("text:")
            print('\n'.join([''.join(line) for line in text]))
            print()

            # Complete short lines with empty spaces to get a rectangular matrix
            max_len = max(len(line) for line in text)
            complete_text = [line + [' '] * (max_len - len(line)) for line in text]
            text = np.array(complete_text)

            # Create a work queue with the grids
            grid_queue = multiprocessing.JoinableQueue()
            for idx, grid in enumerate(cardan_grids):
                grid_queue.put((idx, grid))
            
            # Create a pool of processes to process the grids
            for _ in range(multiprocessing.cpu_count()):
                grid_queue.put((-1, None))
                p = multiprocessing.Process(target=worker, args=(displayed_results_set, output_lock, grid_queue, text, possible_angles, total_grids))
                processes.append(p)
                p.start()

            # Wait until all tasks are done
            grid_queue.join()

            # Wait for each process to finish
            for p in processes:
                p.join()
