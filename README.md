# AI-Powered Thematic Analysis of Music Therapy Sessions
A Thematic Analysis powered by AI tools

## Project Overview

This project utilizes Natural Language Processing (NLP) techniques to analyze transcripts from music therapy group sessions with Black women. It aims to identify, interpret, and visualize the core themes emerging from these sessions. The process involves:

1.  **Topic Modeling (LDA):** Applying Latent Dirichlet Allocation to tags and subtags (derived from transcripts, initially processed using `requalify.ai`) to discover underlying keyword groups.
2.  **Language Model Fine-tuning:** Training a Meta-Llama-3.1-8B model (using the efficient Unsloth library) on the LDA-generated keywords and topic descriptions to create a model capable of interpreting these specific themes.
3.  **AI-driven Theme Interpretation:** Using the fine-tuned Llama model to generate meaningful, context-aware names and descriptions for the identified topics based on their keywords.
4.  **Dimensionality Reduction & Clustering:** Employing PCA and KMeans to visualize the relationships between the topics based on their TF-IDF representations.

The primary script for executing this entire pipeline is `ai_thematic_analysis.py`.

*(Note: A Jupyter Notebook, `TCC_Alice.ipynb`, is present in the repository, reflecting the development process, but the core execution logic resides in the `ai_thematic_analysis.py` script).*

---

## Files in Repository

*   `ai_thematic_analysis.py`: The main Python script containing the analysis pipeline.
*   `resultados_tópicos_com_palavras2.xlsx`: **Required input file** containing the initial data with descriptions, tags, and subtags.
*   `topicos_musicoterapia.json`: Example JSON file formatted for fine-tuning the language model. *Note: The script currently regenerates this.*
*   `Dispersion.png` (or similar): Image file showing initial corpora dispersion (for context, not used by the script).
*   `README.md`: This file.

---

## Prerequisites

### Environment:
*   Python 3.11

### Hardware:
*   **GPU with CUDA:** **Strongly recommended** (ideally required) for the language model fine-tuning step. Training on CPU will be extremely slow. Google Colab's free GPU tier is a viable option.

### Data:
*   The input file `resultados_tópicos_com_palavras2.xlsx` must be present in the same directory as the script, or the `file_path` variable within the script needs to be updated.

---

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder-name>
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   Install `unsloth` and its specific dependencies first, as it handles optimized libraries:
        ```bash
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        pip install --no-deps "xformers" "trl<0.9.0" peft accelerate bitsandbytes transformers # Ensure transformers is installed too
        ```
    *   Install other required libraries:
        ```bash
        pip install pandas openpyxl scikit-learn nltk matplotlib torch 
        ```

4.  **Download NLTK Stopwords:** Run this command in your Python environment or add the necessary lines to the script if not already present:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

5.  **Google Drive Setup (If using Colab or saving model to Drive):**
    *   The script includes steps to fine-tune and save a model. By default (as shown in the provided code), it attempts to save to Google Drive.
    *   If running locally and wanting to save elsewhere, modify the `output_dir` variable for the fine-tuned model in the script.
    *   If running in Colab, ensure the Google Drive mounting code is executed successfully and the specified save path exists in your Drive.

---

## Running the Analysis

1.  **Navigate to Directory:** Open your terminal or command prompt and navigate to the cloned repository directory.
2.  **Activate Virtual Environment (if used):** `source venv/bin/activate`
3.  **Execute the Script:**
    ```bash
    python ai_thematic_analysis.py
    ```

### Script Workflow:

The script will execute the following steps sequentially:

*   **LDA Analysis:** Reads the input Excel, performs LDA, prints the output path for the augmented Excel file (`resultados_tópicos_com_palavras3.xlsx`).
*   **Data Preparation:** Creates the JSON file (`topicos_musicoterapia.json`) for training.
*   **(GPU Intensive) Model Fine-tuning:** Downloads the base Llama 3.1 model, applies LoRA adapters, and fine-tunes it on the prepared JSON data. Progress will be logged to the console. Saves the fine-tuned model to the specified `output_dir` (likely Google Drive) and creates a zip archive.
*   **Inference:** Loads the *fine-tuned* model (requires access to where it was saved, e.g., mounted Drive) and runs inference prompts to interpret topic keywords. The interpretations will be printed to the console.
*   **Dimensionality Reduction:** Performs PCA and KMeans, prints Silhouette analysis results to the console, and displays a plot visualizing the clusters (the plot might save as a file or display in a window depending on your environment).

---

## Outputs

*   `resultados_tópicos_com_palavras3_en.xlsx`: Generated Excel file with added LDA keywords and cluster labels.
*   `topicos_musicoterapia_en.json`: Generated JSON training data.
*   `fine_tuned_model.zip`: Compressed fine-tuned model saved to the location specified in the script (check Google Drive path).
*   **Console Output:** Progress indicators, LDA/Clustering results, and AI-generated topic interpretations.
*   **Plot:** A matplotlib window or saved image file showing the PCA/KMeans cluster visualization.

## Notes

*   **GPU Requirement:** The model fine-tuning part **needs a capable GPU** for feasible execution times.
*   **File Paths:** You may need to **adjust file paths** inside `ai_thematic_analysis.py` (for input Excel and model saving/loading) depending on where you place the files and your operating system.
*   **Language:** The fine-tuned model expects and generates interpretations based on the *Portuguese* instructions and keywords it was trained on. The surrounding script logic and comments are in English.
*   **Colab:** Running this script within Google Colab (with a GPU runtime) simplifies dependency management and provides free GPU access. Remember to mount your Google Drive if saving/loading the model there.

## Author and Contact

Frederico Pedrosa  
fredericopedrosa@ufmg.br
