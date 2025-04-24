# -*- coding: utf-8 -*-

#0. The notebook is structured as follows:

# 1. Dispersion of the *corpora* and performing LDA on tags and subtags identified by requalify.ai
# 2. Language model training;
# 3. Using the fine-tuned model to discover the themes of each group
# 4. Dimensionality reduction of tags and subtags.

#2.0. Dispersion graph of the *corpora* made by Voyant Tools
# ![Dispersion.png](data:image/png;base64,...) # Image markdown kept for context, but won't render directly in code

# The themes are very clustered. One solution was to perform a survey of tags and subtags using requalify.ai.
# Content of the file resultados tópicos com palavras 3.xlsx

#1. LDA of the tags and subtags identified by requalify.ai
# """ # Original comment block end

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import torch # Added import for torch used later

# Load stopwords in Portuguese
nltk.download('stopwords')

# Load the Excel file
file_path = "resultados_tópicos_com_palavras2.xlsx" # Original file name kept
sheet_name = "Planilha1" # Original sheet name kept
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Select the descriptions column
descriptions = df['Description'].dropna()

# Pre-process the text with Portuguese stopwords
stop_words_pt = stopwords.words('portuguese')  # List of stopwords in Portuguese
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words_pt)
X = vectorizer.fit_transform(descriptions)

# Apply LDA for topic modeling
num_topics = len(descriptions)  # One topic per description
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(X)

# Extract the most important words per topic
feature_names = vectorizer.get_feature_names_out()
top_words_per_topic = []

for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]  # Top 10 words per topic
    top_words_per_topic.append(", ".join(top_words))

# Add the most important words as a new column in the DataFrame
df['Main Topic'] = top_words_per_topic # Translated column name

# Save the results to a new Excel file (optional)
output_path = "resultados_tópicos_com_palavras3_en.xlsx" # Adjusted output filename
df.to_excel(output_path, index=False)
print(f"Results saved to: {output_path}")

# """_____ # Original separator comment
#2. Training the Meta-llama 3.1 7b model

### Check stopwords
# """ # Original comment block end

# Download NLTK stopwords
nltk.download('stopwords')

# Get the list of stopwords in Portuguese
stop_words_pt = stopwords.words('portuguese')

print("Stopwords in Portuguese:") # Translated print description
print(stop_words_pt)

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers" "trl<0.9.0" peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# """### Data Preparation""" # Original comment block end

import pandas as pd
import json

# Load the Excel file
file_path = "/content/drive/MyDrive/UFMG/Alice/resultados_tópicos_com_palavras3.xlsx" # Original file name kept
df = pd.read_excel(file_path)

# Create the JSON structure
data = []
for _, row in df.iterrows():
    topic = row['Topic']
    description = row['Description']
    # The terms column contains Portuguese words relevant to the model's task
    # These should NOT be translated per the request.
    terms = row['Terms'].split(", ")  # Separate keywords

    entry = {
        "instruction": ( # Translated instruction
            "Analyze the provided words and name the topics, interpreting the main meaning "
            "of each group of words, considering that it is a transcription "
            "of music therapy groups with Black women."
        ),
        "input": f"Here are the words: {terms}.", # Input keeps the Portuguese words
        "output": f"{topic}. Descrição: {description}" # Output keeps "Descrição:"
    }
    data.append(entry)

# Save as JSON
output_path = "topicos_musicoterapia_en.json" # Adjusted output filename
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"JSON file saved at: {output_path}") # Translated print statement

custom_dataset = True # Renamed variable for clarity

if custom_dataset:
  alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Input:
  {}

  ### Response:
  {}""" # Kept original English prompt

  EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
  def formatting_prompts_func(examples):
      instructions = examples["instruction"]
      inputs       = examples["input"] # Corrected variable name from original
      outputs      = examples["output"]
      texts = []
      for instruction, input_text, output in zip(instructions, inputs, outputs): # Adjusted loop for input
          # Must add EOS_TOKEN, otherwise your generation will go on forever!
          text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN # Adjusted format call
          texts.append(text)
      return { "text" : texts, }
  pass

  from datasets import load_dataset
  dataset = load_dataset("json", data_files="topicos_musicoterapia_en.json", split = "train") # Adjusted filename
  dataset = dataset.map(formatting_prompts_func, batched = True,)

else:
  # This part was not in the original execution path but kept for completeness
  from datasets import load_dataset
  dataset = load_dataset("vicgalle/alpaca-gpt4", split = "train")
  print(dataset.column_names)

print(dataset.column_names)

# The following section seems to attempt alternative formatting/templating.
# It will be translated but might need adjustment depending on the desired final format.

# from unsloth import to_sharegpt
# dataset = to_sharegpt(
#     dataset,
#     merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
#     output_column_name = "output",
#     conversation_extension = 3,
# )

# from unsloth import standardize_sharegpt
# dataset = standardize_sharegpt(dataset)

# Translated Alpaca Prompt (as used later in inference)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Translated Chat Template (as used later in inference)
chat_template = """Below are some instructions describing tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template

# Customize the default_system_message for chatbot behavior (Translated)
default_system_message = (
    "You are an empathetic and knowledgeable assistant, trained to analyze and interpret topics "
    "from music therapy sessions with Black women. Your goal is to provide insightful "
    "and contextually appropriate responses."
)

# Using the `text` field generated by `formatting_prompts_func` for SFTTrainer
# The apply_chat_template step might be redundant if the previous formatting is sufficient,
# or it might need adjustment if a chat-specific format is desired *after* the initial formatting.
# For now, let's assume the `formatting_prompts_func` output is what SFTTrainer expects.
# If `apply_chat_template` is needed, it should operate on appropriate columns.
# Commenting out apply_chat_template as its input structure seems incompatible with the prior step.
# dataset = apply_chat_template(
#     dataset,
#     tokenizer=tokenizer,
#     chat_template=chat_template, # Uses the English template defined above
#     default_system_message=default_system_message
# )


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # Uses the 'text' field from formatting_prompts_func
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        #num_train_epochs = 60, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show memory status # Translated title
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# """###Training""" # Original comment block end

trainer_stats = trainer.train()

#@title Show final memory and training status # Translated title
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

import shutil
output_dir = "/content/drive/MyDrive/UFMG/Alice/fine_tuned_model" # Adjusted dir name

# Save the model in Hugging Face format
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Compress the folder to save space
shutil.make_archive(output_dir, 'zip', output_dir)

print(f"Model saved and compressed at: {output_dir}.zip") # Translated print statement

# """### Save model to local storage""" # Original comment block end

# Save to 8bit Q8_0
if True: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("modelQ4", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )

import torch
print(torch.cuda.is_available())

# """____
# #4. Using the fine-tuned model to discover the themes of each group

# It is necessary to call the model using float16.
# """ # Original comment block end

from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the saved model
model_dir = "/content/drive/MyDrive/UFMG/Alice/fine_tuned_model" # Adjusted dir name

# Load the model with float16
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.float16  # Force the use of float16
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

print("Model loaded with float16!") # Translated print statement

# Need to reinstall unsloth if runtime was restarted or packages were uninstalled
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install transformers==4.47.1 torch==2.5.1 # Version pinning might be needed depending on environment changes

# """##Group 1 - Topic 1""" # Original comment block end

from transformers import TextStreamer, DynamicCache
from unsloth import FastLanguageModel

# Configuration to use or not use the chat template
use_chat_template = True # Renamed variable

if use_chat_template:
    # Activate optimized inference
    FastLanguageModel.for_inference(model)  # Activate 2x faster inference

    # Messages in chat format
    messages = [
        {"role": "user", "content": "Analise as palavras fornecidas e dê o nome ao tópico, interpretando o significado principal do grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras."} # Portuguese instruction kept for the *fine-tuned* model
    ]

    # Apply the chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Streamer for text generation
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Initialize dynamic cache (new API)
    past_key_values = DynamicCache()

    # Generate text with dynamic cache
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

else:
    # Prompt in Alpaca format
    FastLanguageModel.for_inference(model)  # Activate 2x faster inference

    # Define the Alpaca prompt (using the translated English one here, but the fine-tuned model expects the Portuguese one used during training)
    # Using the Portuguese one for consistency with training data structure
    alpaca_prompt = """Abaixo estão algumas instruções que descrevem tarefas. Escreva respostas que completem adequadamente cada solicitação.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # List of words (topics) - KEEP PORTUGUESE WORDS
    topic_words = "['então', 'só', 'falou', 'pessoas', 'acho', 'ia', 'fica', 'queria', 'difícil', 'vamos']"

    # Create inputs for the model based on the prompt
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Analise as palavras fornecidas e dê o nome aos tópicos, interpretando o significado principal de cada grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras.",  # Instruction (Portuguese)
                f"Aqui estão as palavras: {topic_words}",  # Input (Portuguese words)
                "",  # Empty output for generation!
            )
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")

    # Explicitly add attention mask
    inputs["attention_mask"] = inputs["input_ids"] != tokenizer.pad_token_id

    # Streamer for text generation
    text_streamer = TextStreamer(tokenizer)

    # Initialize dynamic cache (new API)
    past_key_values = DynamicCache()

    # Generate text with dynamic cache
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# """##Group 1 - Topic 2""" # Original comment block end

use_chat_template = True

if use_chat_template:
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "user", "content": "Analise as palavras fornecidas e dê o nome ao tópico, interpretando o significado principal do grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras."} # Portuguese instruction
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    past_key_values = DynamicCache()
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

else:
    FastLanguageModel.for_inference(model)
    alpaca_prompt = """Abaixo estão algumas instruções que descrevem tarefas. Escreva respostas que completem adequadamente cada solicitação.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # KEEP PORTUGUESE WORDS
    topic_words = "['vou', 'gente', 'falei', 'tinha', 'coisas', 'semana', 'fiquei', 'tava', 'trabalhar', 'saúde']"

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Analise as palavras fornecidas e dê o nome aos tópicos, interpretando o significado principal de cada grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras.",  # Instruction (Portuguese)
                f"Aqui estão as palavras: {topic_words}", # Input (Portuguese words)
                "",
            )
        ],
        return_tensors="pt", padding=True, truncation=True,
    ).to("cuda")
    inputs["attention_mask"] = inputs["input_ids"] != tokenizer.pad_token_id
    text_streamer = TextStreamer(tokenizer)
    past_key_values = DynamicCache()
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# """## Group 1 - Topic 3""" # Original comment block end

use_chat_template = True

if use_chat_template:
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user", "content": "Analise as palavras fornecidas e dê o nome ao tópico, interpretando o significado principal do grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras."} # Portuguese instruction
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    past_key_values = DynamicCache()
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

else:
    FastLanguageModel.for_inference(model)
    alpaca_prompt = """Abaixo estão algumas instruções que descrevem tarefas. Escreva respostas que completem adequadamente cada solicitação.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # KEEP PORTUGUESE WORDS
    topic_words = "['gente', 'então', 'falar', 'música', 'só', 'tinha', 'fala', 'coisas', 'mãe', 'mesmo']"

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Analise as palavras fornecidas e dê o nome aos tópicos, interpretando o significado principal de cada grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras.",  # Instruction (Portuguese)
                f"Aqui estão as palavras: {topic_words}", # Input (Portuguese words)
                "",
            )
        ],
        return_tensors="pt", padding=True, truncation=True,
    ).to("cuda")
    inputs["attention_mask"] = inputs["input_ids"] != tokenizer.pad_token_id
    text_streamer = TextStreamer(tokenizer)
    past_key_values = DynamicCache()
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# """##Group 2 - Topic 1""" # Original comment block end

use_chat_template = True

if use_chat_template:
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user", "content": "Analise as palavras fornecidas e dê o nome ao tópico, interpretando o significado principal do grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras."} # Portuguese instruction
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    past_key_values = DynamicCache()
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

else:
    FastLanguageModel.for_inference(model)
    alpaca_prompt = """Abaixo estão algumas instruções que descrevem tarefas. Escreva respostas que completem adequadamente cada solicitação.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # KEEP PORTUGUESE WORDS
    topic_words = "['música', 'falar', 'mulher', 'acho', 'mim', 'vida', 'ia', 'mesmo', 'dela', 'queria']"

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Analise as palavras fornecidas e dê o nome aos tópicos, interpretando o significado principal de cada grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras.",  # Instruction (Portuguese)
                f"Aqui estão as palavras: {topic_words}", # Input (Portuguese words)
                "",
            )
        ],
        return_tensors="pt", padding=True, truncation=True,
    ).to("cuda")
    inputs["attention_mask"] = inputs["input_ids"] != tokenizer.pad_token_id
    text_streamer = TextStreamer(tokenizer)
    past_key_values = DynamicCache()
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# """##Group 2 - Topic 2""" # Original comment block end

use_chat_template = True

if use_chat_template:
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user", "content": "Analise as palavras fornecidas e dê o nome ao tópico, interpretando o significado principal do grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras."} # Portuguese instruction
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    past_key_values = DynamicCache()
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

else:
    FastLanguageModel.for_inference(model)
    alpaca_prompt = """Abaixo estão algumas instruções que descrevem tarefas. Escreva respostas que completem adequadamente cada solicitação.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # KEEP PORTUGUESE WORDS
    topic_words = "['vou', 'então', 'tinha', 'entendeu', 'dele', 'mulheres', 'fiquei', 'tocar', 'situação', 'vejo']"

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Analise as palavras fornecidas e dê o nome aos tópicos, interpretando o significado principal de cada grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras.",  # Instruction (Portuguese)
                f"Aqui estão as palavras: {topic_words}", # Input (Portuguese words)
                "",
            )
        ],
        return_tensors="pt", padding=True, truncation=True,
    ).to("cuda")
    inputs["attention_mask"] = inputs["input_ids"] != tokenizer.pad_token_id
    text_streamer = TextStreamer(tokenizer)
    past_key_values = DynamicCache()
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# """##Group 2 - Topic 3""" # Original comment block end

use_chat_template = True

if use_chat_template:
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user", "content": "Analise as palavras fornecidas e dê o nome ao tópico, interpretando o significado principal do grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras."} # Portuguese instruction
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    past_key_values = DynamicCache()
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

else:
    FastLanguageModel.for_inference(model)
    alpaca_prompt = """Abaixo estão algumas instruções que descrevem tarefas. Escreva respostas que completem adequadamente cada solicitação.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # KEEP PORTUGUESE WORDS
    topic_words = "['gente', 'só', 'coisas', 'falei', 'fala', 'falou', 'então', 'pessoas', 'tava', 'ficar']"

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Analise as palavras fornecidas e dê o nome aos tópicos, interpretando o significado principal de cada grupo de palavras considerando que se trata de uma transcrição de grupos de musicoterapia com mulheres negras.",  # Instruction (Portuguese)
                f"Aqui estão as palavras: {topic_words}", # Input (Portuguese words)
                "",
            )
        ],
        return_tensors="pt", padding=True, truncation=True,
    ).to("cuda")
    inputs["attention_mask"] = inputs["input_ids"] != tokenizer.pad_token_id
    text_streamer = TextStreamer(tokenizer)
    past_key_values = DynamicCache()
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# """___

# #5. Dimensionality reduction of tags
# """ # Original comment block end

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples # Added imports for silhouette analysis

# Load topic data
file_path = "/content/drive/MyDrive/UFMG/Alice/resultados_tópicos_com_palavras3_en.xlsx" # Adjusted filename
df = pd.read_excel(file_path)

# Select columns containing the complete topics
topics = df['Topic'].tolist() # Assumes 'Topic' column contains the full topic descriptions

# Vectorize topics using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(topics)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

# Use KMeans++ method to improve centroid positions
kmeans_plus_plus = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
kmeans_plus_plus.fit(X_pca)

# Plot the data
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

# Plot the centroids
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

# Plot decision boundaries
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

plt.figure(figsize=(8, 6))
plot_decision_boundaries(kmeans_plus_plus, X_pca)
plt.show()

# Initialize a dictionary to store silhouette scores
silhouette_scores = {}
kmeans_per_k = {}

# Fit KMeans models for k between 2 and 20 and calculate silhouette coefficient
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init='auto' or 10
    kmeans.fit(X_pca)  # Fit the model
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores[k] = score
    kmeans_per_k[k] = kmeans
    print(f"Silhouette Coefficient for k={k}: {round(score, 3)}")

# """#####The coefficient with k = 3 has the highest value. Using k = 3 was appropriate and correctly classified 97.65% of the data.""" # Original comment block end - Translated

# Calculate silhouette coefficients for the last k value (k=3 was identified as best)
k = 3
kmeans = kmeans_per_k[k]
silhouette_coefficients = silhouette_samples(X_pca, kmeans.labels_)

# Count how many silhouette coefficients were below 0, i.e., classification errors for k=3
classification_errors_indices = np.where(silhouette_coefficients < 0)[0]
classification_errors = len(classification_errors_indices)
percentage = classification_errors / len(silhouette_coefficients) if len(silhouette_coefficients) > 0 else 0
print(f"Number of classification errors: {classification_errors} ({percentage:.2%})") # Translated print statement

# """####The texts grouped in each cluster were:""" # Original comment block end - Translated

# Get cluster labels
cluster_labels = kmeans_plus_plus.labels_ # Use the k=3 model

# Add cluster labels to the original DataFrame
df['Cluster'] = cluster_labels

# Group topics by cluster
clusters = df.groupby('Cluster')['Topic'].apply(list)

# Display topics in each cluster
for cluster, topic_list in clusters.items(): # Renamed variable
    print(f"Cluster {cluster}:") # Translated print statement
    for topic_item in topic_list: # Renamed variable
        print(f" - {topic_item}")
    print("\n")

# Identify points that were classification errors (silhouette coefficients below 0)
# classification_errors_indices already calculated above
classification_errors_points = [topics[i] for i in classification_errors_indices]

# Display the points that were classification errors
print("Points that were classification errors:") # Translated print statement
for point in classification_errors_points:
    print(point)

!python --version
