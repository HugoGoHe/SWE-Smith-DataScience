import modal

alpaca_prompt = """Below is an instruction that describes a task, paired with an incorrect input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


HF_TOKEN="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"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
N_GPU = 1

import modal

import modal

finetune_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    # 1️⃣ Instalar dependencias del sistema necesarias para compilar llama.cpp con CMake (incluyendo libcurl)
    .apt_install(
        "git",
        "cmake",
        "build-essential",
        "python3-dev",
        "libomp-dev",
        "libopenblas-dev",
        "libcurl4-openssl-dev",  # Dependencia para que CMake encuentre CURL
    )
    # 2️⃣ Clonar llama.cpp y compilar con CMake + copiar ejecutables al directorio raíz de llama.cpp
    .run_commands([
        # Clonar el repositorio de llama.cpp
        "git clone https://github.com/ggerganov/llama.cpp /root/llama.cpp",
        # Usar CMake para generar la carpeta build
        "cd /root/llama.cpp && cmake -B build",
        # Construir en modo Release
        "cd /root/llama.cpp && cmake --build build --config Release -j\"$(nproc)\"",
        # Copiar los ejecutables (llama-*) de build/bin al directorio raíz de llama.cpp
        "cp /root/llama.cpp/build/bin/llama* /root/llama.cpp/"
    ])
    # 3️⃣ Instalar librerías Python necesarias para entrenamiento
    .pip_install(
        "cupy-cuda12x",
        "torch",
        "transformers>=4.51.0",
        "unsloth",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5"
    )
    .pip_install("datasets==3.6.0", "huggingface_hub", "hf_transfer")
    # 4️⃣ Activar variable de entorno para optimizar transferencias de modelo
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)



hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)




app = modal.App("fine-tuning-some-model")

@app.function(
    image=finetune_image,
    gpu=f"L40S{N_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=4 * HOURS,
)
def finetune():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch
    import datasets

    from datasets import load_dataset
    
    
    print("Downloading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-Coder-14B-Instruct",
        max_seq_length=32768,
        dtype = None,
        load_in_4bit=True,
        load_in_8bit=False,
    )
    
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )
    
    


    ALLOWED_REPOS = [
        "alethomas/voluptuous",
        "andialbrecht/sqlparse",
        "buriiy/python-readability",
        "burnash/gspread",
        "chardet/chardet",
        "cloudpipe/cloudpickle",
        "dask/dask",
        "datamade/usaddress",
        "davidhalt/parso",
        "erikrose/parsimonious",
        "facelessuser/soupsieve",
        "gawel/pyquery",
        "google/textfsm",
        "gruns/furl",
        "gweis/isodate",
        "hukkin/tomli",
        "jawah/charset_normalizer",
        "john-kurkowski/tldextract",
        "joke2k/faker",
        "jsvine/pdfplumber",
        "kayak/pypika",
        "keleshev/schema",
        "kennethreitz/records",
        "kurtmckee/feedparser",
        "leptrure/mistune",
        "madzak/python-json-logger",
        "mahmoud/glom",
        "marshmallow-code/marshmallow",
        "martinblech/xmldict",
        "matthewwithanm/python-markdownify",
        "mewwts/addict",
        "mido/mido",
        "modin-project/modin",
        "mozilla/bleach",
        "msiemens/tinydb",
        "pandas-dev/pandas",
        "pdfminer/pdfminer.six",
        "pudo/dataset",
        "pydantic/pydantic",
        "pydata/patsy",
        "pydicom/pydicom",
        "pygments/pygments",
        "pyparsing/pyparsing",
        "python-jsonschema/jsonschema",
        "python-openxml/python-docx",
        "r1chardj0n3s/parse",
        "scanny/python-pptx",
        "scrapy/scrapy",
        "seperman/deepdiff",
        "sloria/environs",
        "sunpy/sunpy",
        "tkrajina/gpxpy",
        "tobymao/sqlglot",
        "un33k/python-slugify",
    ]


    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]  # each “convo” is a list of dicts
        
        texts = []
        for convo in convos:
            # 1) Drop all system‐role messages:
            # filtered = [m for m in convo if m.get("role") != "system"]
            
            # 2) Now pass only user/assistant messages into apply_chat_template:
            wrapped = tokenizer.apply_chat_template(
                filtered,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(wrapped)
        
        return {"text": texts}

    from datasets import Dataset
    # Perform the filter
    filtered = load_dataset("Daniel4190/filtered_models_swe_smith")
    filtered = filtered["train"]
    print("Filtered: ", filtered)
    filtered = [D["messages"] for D in filtered]
    filtered = Dataset.from_dict({"conversations":filtered})
    print("Filtered: ", filtered)
    filtered = filtered.map(formatting_prompts_func, batched = True)
    print("Conversations: ", repr(filtered["text"][0]))


    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    

   
    

  #  from datasets import load_dataset

 #   print("Downloading and splitting datasets...")
 #   original_dataset = load_dataset("Daniel4190/filtered_models_swe_smith")
    
    #from unsloth.chat_templates import standardize_sharegpt

    #formatted_dataset = standardize_sharegpt(filtered)


    #formatted_dataset = original_dataset.map(formatting_prompts_func, batched=True)
    #train_test_split_dataset = formatted_dataset["train"].train_test_split(test_size=0.15)

    #train_and_test_AGAIN = train_test_split_dataset["train"].train_test_split(test_size=0.15)

    # train, test and eval
    #test_dataset = train_test_split_dataset["test"]
    #train_dataset = train_and_test_AGAIN["train"]
    #eval_dataset = train_and_test_AGAIN["test"]


    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported
    
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = filtered,
    dataset_text_field = "text",
    max_seq_length = 32768,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8, # Fixed major bug in latest Unsloth
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = None,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit", # Save more memory
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        ),
    )
    print("Dataset: ", filtered)
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    trainer_stats = trainer.train()
    
    
    
    
    if True: model.push_to_hub_gguf("REPO_NAME", tokenizer, token = "SECRET_KEY")


