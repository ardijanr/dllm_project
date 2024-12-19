import json
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf
from huggingface_hub import login
import time
import csv
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import os




login("HF TOKEN PASTED HERE")


MODEL_NAME = "MODEL_NAME"
MODEL_NAME = "openai-community/gpt2-medium"
INPUT_TEXT = "Write me the first paragraph which comes to mind."
MAX_LENGTH = 200
TEMPERATURE = 0.7
OUTPUT_FILE = "native_inference.csv"
DRY_RUN_DURATION_MINUTES = 30
MODEL_CACHE_DIR = "model_cache"
SAVED_MODEL_DIR = MODEL_CACHE_DIR+"/openai-community--gpt2-medium-tfjs"

print("PRINTING GPUS!")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print(gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

print("Starting model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
tokens = tokenizer.encode(INPUT_TEXT)

encoded_inputs = tokenizer(INPUT_TEXT, return_tensors="pt")

# Save tokens and attention_mask to a JSON file
data = {
    "input_ids": encoded_inputs["input_ids"].tolist()[0],
    "attention_mask": encoded_inputs["attention_mask"].tolist()[0],
}

encoded_inputs = tokenizer(INPUT_TEXT, return_tensors="tf", truncation=True)


with open(SAVED_MODEL_DIR+"/token_and_mask.json", "w") as f:
    f.write(json.dumps(data))
    f.close()



print("LOADING MODEL")
model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME, from_pt=True, cache_dir=MODEL_CACHE_DIR)


model.save_pretrained(SAVED_MODEL_DIR, saved_model=True)
tokenizer.save_pretrained(SAVED_MODEL_DIR)
inputs = tokenizer(INPUT_TEXT, return_tensors="tf")


print("Starting 30-minute warmup inference loop...")

end_time = time.time() + DRY_RUN_DURATION_MINUTES * 0
while time.time() < end_time:
    print("RUNNING WARMUP")
    with tf.device('/GPU:0'):

        model.generate(
            input_ids=encoded_inputs["input_ids"],
            attention_mask=encoded_inputs["attention_mask"],
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

print("Finished 30-minute warmup inference loop.")


with open(OUTPUT_FILE, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Execution Time (seconds)"])

    for i in range(100):
        with tf.device('/GPU:0'):
            start_time = time.time()
            model.generate(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
                max_length=MAX_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            execution_time = time.time() - start_time
            print(f"Running test iteration {i} took {execution_time}")

            csvwriter.writerow([execution_time])
    csvfile.close()

print(f"Execution times saved to {OUTPUT_FILE}.")
