# Cardan Decrypt

### Overview
**Cardan Decrypt** is a Python-based decryption tool for texts encrypted with the **Cardan Grille cipher**. The project generates or loads **square Cardan grids** of different sizes for decryption and supports **non-square manual grids** provided in `input_grid.txt`. Additional features include **Caesar cipher rotations**, **segmentation validation** with a word dictionary, and **grid and text transformations** to improve decryption accuracy.

The tool uses **multithreading** to handle grid generation and decryption attempts concurrently, improving performance and making efficient use of system resources. **Grids are generated on the fly** as they are processed, rather than being stored in memory, allowing for large-scale decryption tasks without high memory demands.

---

## Parameters and Configurations in `cardan.py`

### Grid and Rotation Configurations
- **`grid_size`**: Sets the size of the generated square Cardan grids. The custom grid provided by `input_grid.txt` does not have to be square and is processed first. The grids are generated on the fly as they are processed (not stored all in memory at once).
  
- **`possible_angles`**: Defines rotation sequences for grids during decryption attempts. Each sequence in `possible_angles` is a list of four integers, each representing the angle (in degrees) by which the grid is rotated for each pass over the text. Here are some sample configurations:
  - `[0, 0, 0, 0]`: No rotation, grid is used as-is.
  - `[0, 180, 0, 180]`: Alternates between no rotation and 180 degrees.
  - `[0, 90, 180, 270]`: Rotates the grid clockwise by 90 degrees with each pass.
  - `[0, 270, 180, 90]`: Rotates the grid in a different order, enhancing decryption flexibility.

The tool applies each sequence in `possible_angles` to maximize coverage, testing various grid orientations for better decryption accuracy.

**Note**: Cardan grids containing full rows of `1`s are ignored, as these do not enable effective encryption.

### Additional Configurations

- **`percentage_threshold`**: Sets the minimum percentage of recognized words required in the decrypted text for it to be considered valid. For example, `percentage_threshold = 97` means that 97% of the decrypted text must be valid words.
- **`word_size_threshold`**: Specifies the minimum average word length in the decrypted text for validation. If the average word length is below this threshold, the decryption is ignored. Example: `word_size_threshold = 3.6`.
- **`caesar_num_rotations`**: Defines the number of Caesar cipher rotations to apply to the decrypted text. A value of `0` means no Caesar rotation, while other values specify the number of shifts.
- **`dictionary_path`**: Sets the path to the word dictionary file, used to validate segmented words in the decrypted text. Example path: `dictionary_path = os.path.join(project_dir, 'data', 'words_wiktionary_fr.txt')`.
- **`exceptions_file`**: Specifies the path to a file containing exception words to ignore during validation. Example path: `exceptions_file = os.path.join(project_dir, 'data', 'words_exceptions.txt')`.
- **`worker_size`**: Sets the number of worker processes to handle grid processing. It is set by default to `multiprocessing.cpu_count() - 1`, meaning it will utilize all but one of the available CPU cores, allowing some resources to remain available for other tasks.
- **`max_grid_queue_size`**: Limits the maximum number of grids in the queue at a given time to control memory usage during multithreaded processing. The default is set to `1000`, but this can be adjusted based on system memory and performance requirements.

### Grid and Text Transformations
During each decryption attempt, multiple transformations are applied to both the **grid** and the **text**:

- **Reverse Loops**: Reverses the order in which the grid traverses the text, changing the reading direction.
- **Reset Angle Index**: Resets the angle index at the start of each new line, affecting how the grid rotates across the text.
- **Text Reversal**: After processing each text layout, both the original and reversed (backwards) versions of the decrypted text are checked for potential matches.

### Text Configurations and Transformations
To improve decryption accuracy, the tool generates alternative configurations of the input text by removing spaces and various punctuation marks, as well as reshaping the text into square matrices when possible. These transformations are applied as follows:

1. **Progressive Cleaning**: The tool creates multiple versions of the text by removing different sets of characters:
   - **Original text**: Used without modification.
   - **Spaces removed**: All spaces in the text are removed.
   - **Spaces, commas, and periods removed**: The text is cleaned of spaces, commas, and periods.
   - **Spaces, commas, periods, and apostrophes removed**: The text is cleaned of spaces and additional punctuation like commas, periods, and apostrophes.

2. **Square Reshaping**: For each cleaned text version, the tool attempts to reshape the text into the largest possible square matrix (if the character count allows). This process involves rearranging the text into a grid-like format, increasing compatibility with the Cardan grids.

3. **Unique Configurations**: After generating these cleaned and reshaped versions, the tool filters out any duplicate configurations to avoid redundant decryption attempts.

---

## Example Workflow

For each unique **text layout**, the following decryption steps are performed:
1. **Apply the Grid**: Each layout is processed with the grid in each configuration of `possible_angles`.
2. **Reverse Loops** and **Reset Angle Index**: These transformations are applied as the grid traverses the text.
3. **Text Reversal**: After applying the grid, both the forward and reversed forms of the decrypted text are checked.
4. **Segmentation Validation**: Each resulting decrypted text is validated for segmentation accuracy against the word dictionary, considering **percentage threshold** and **word size threshold**.

---

## Installation

### Requirements
- **Python 3.7+**
- Install dependencies:

  ```bash
  pip install numpy requests
  ```

### Clone the Repository
```bash
git clone https://github.com/algofrench/cardan_decrypt.git
cd cardan_decrypt
```

### Download and Parse Dictionary (Optional)
To improve word segmentation and validation accuracy, you can run `create_dict.py` to download the latest Wiktionary dump and generate dictionary files:

```bash
python script/create_dict.py
```
The script downloads `frwiktionary-latest-pages-articles.xml.bz2` to the `data/` directory and generates `words_wiktionary_fr.txt` and `words_wiktionary_fr.csv`. Running this script is optional but recommended for projects requiring a robust word dictionary.

### Configuration for Dictionary Generation

In `create_dict.py`, you can configure the language setting to generate a dictionary for different languages. By default, the language is set to French (`language = "fr"`), but you can adjust this setting to any other supported language code.

**Example Configuration in `create_dict.py`:**

```python
language = "fr"  # Set the language for dictionary generation (e.g., "en" for English, "es" for Spanish)
```

---

## Usage

### Run Decryption
Run `cardan.py` to decrypt the text:

```bash
python script/cardan.py
```
This will automatically attempt all transformations and configurations, outputting any decrypted text segments that match the configured word validation thresholds.

### Example Command
The `cardan.py` script reads `input_grid.txt` (if provided) and `input_text.txt`, then applies each grid and text transformation in sequence to maximize decryption possibilities.

---

## Example of Cardan Decrypt in Action

This example demonstrates the use of **Cardan Decrypt** with sample text and a specific Cardan grid configuration to showcase the transformation steps, grid applications, and final decrypted output.

### Input Files and Configuration

For this example, the following inputs and configurations are used:

- **Content of `input_text.txt`**:
```bash
Boulangère sottement osée, sucrant deposant transie, ile localisée, agile saucière régulée.
```

- **Content of `input_grid.txt`**:
```bash
0 0 0 1
0 1 1 0
1 0 0 0
0 1 1 0
```

- **Configuration in `cardan.py`**:

```python
grid_size = 2  # Size of the square grids
percentage_threshold = 97  # Threshold for the percentage of recognized words
word_size_threshold = 3.6  # Minimum average word size to consider a decryption valid
possible_angles = [[0, 0, 0, 0], [0, 180, 0, 180], [0, 90, 180, 270], [0, 270, 180, 90]]  # Rotation configurations for grid alignment
caesar_num_rotations = 0  # Number of Caesar cipher shifts applied to the decrypted text
dictionary_path = os.path.join(project_dir, 'data', 'words_wiktionary_fr.txt')  # File containing the word dictionary
exceptions_file = os.path.join(project_dir, 'data', 'words_exceptions.txt')  # File containing exception words to ignore
grid_file = os.path.join(project_dir, 'input_grid.txt')  # Path to the specific grid used
text_file = os.path.join(project_dir, 'input_text.txt')  # Path to the input text to be decrypted
```

### Output Example

Below is a sample output generated by `cardan.py` using the above inputs, let's analyze it:
```
Number of unique words: 1654560
len(texts): 6

text:
Boulangère sottement osée, sucrant deposant transie, ile localisée, agile saucière régulée.

Cardan Grid 10/10:

text:
Boulangèresottementosée,sucrantdeposanttransie,ilelocalisée,agilesaucièrerégulée.

Cardan Grid 10/10:

text:
Boulangèresottementoséesucrantdeposanttransieilelocaliséeagilesaucièrerégulée

Cardan Grid 10/10:

text:
Boulangèr
e sotteme
nt osée,
sucrant d
eposant t
ransie, i
le locali
sée, agil
e saucièr
e régulée
.

Cardan Grid 10/10:

text:
Boulangèr
esottemen
tosée,suc
rantdepos
anttransi
e,ileloca
lisée,agi
lesaucièr
erégulée.

Cardan Grid 10/10:

text:
Boulangè
resottem
entosées
ucrantde
posanttr
ansieile
localisé
eagilesa
ucièreré
gulée

Cardan Grid 1/10:
0 0 0 1
0 1 1 0
1 0 0 0
0 1 1 0
rot: 0
angles: [0, 0, 0, 0]
string( 28) : lesecretestdanslagrilleseule
Optimal segmentation(100.00%, 4.00) : le secret est dans la grille seule
```

The final decrypted message is:

**<span style="color:red">"le secret est dans la grille seule"</span>**

### Explanation of Output

1. **Word Count and Text Transformations**:
   - `Number of unique words: 1654560` indicates the total number of unique French words available in the dictionary for validation.
   - `len(texts): 6` shows that 6 different text transformations were created from the input text by removing various punctuation or spacing, or reshaping it into square matrices.

2. **Input Text and Transformations**:
   - The input text is: `"Boulangère sottement osée, sucrant deposant transie, ile localisée, agile saucière régulée."`
   - Following transformations are applied to this text to prepare it for decryption:
     - **Transformation 1**: Removes spaces between words while keeping punctuation.
     - **Transformation 2**: Removes both spaces and punctuation to produce a continuous string of characters.
     - **Transformation 3 to 6**: Reshape the text into grids with different alignments for each transformed version.

3. **Application of Cardan Grids**:
   - `Cardan Grid 1/10:` shows the first grid used in the process, with cells arranged as follows:
     ```
     0 0 0 1
     0 1 1 0
     1 0 0 0
     0 1 1 0
     ```
   - This configuration directs the placement of characters in the encrypted message.
   - **Rotation**: `rot: 0` indicates no rotation is applied to the grid.
   - **Angles**: `[0, 0, 0, 0]` specifies the angle rotation sequence for the grid (no rotation in this case).

4. **Decryption Output**:
   - The final output shows `string( 28) : lesecretestdanslagrilleseule`, which is a successful decryption of the original message hidden in the text transformations.
   - **Optimal segmentation**: The text is segmented into words, showing `le secret est dans la grille seule`, indicating a 100% match and an average word length of 4 characters.
   - This demonstrates the tool's ability to extract meaningful phrases hidden within the transformed text using a specified Cardan grid.

This example illustrates how **Cardan Decrypt** processes an input text through transformations, applies Cardan grids, and outputs meaningful decrypted messages.

---

## License
This project is licensed under the Apache License 2.0. See the full text in the [LICENSE](LICENSE) file.
