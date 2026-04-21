import json
import os
import torch
import re
import gc
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,       # LLaVA-1.5
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration    # LLaVA-NeXT
)
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))



GLOBAL_CONFIG = {
    "BASE_DIR": {
        "sim": os.path.join(PROJECT_ROOT, "simdata_benchmark"),
        "real": os.path.join(PROJECT_ROOT, "realdata_benchmark")
    },
    
    
    "DATASETS": {
        "sim": "pure_sim_test.json",
        "real": "pure_real_test.json"
    },

    
    "MODELS": {
        "llava-1.5": {
            "base_path": "llava-1.5-7b-hf",
            "lora_path": os.path.join(PROJECT_ROOT, "LLaMA-Factory/saves/llava-1_5-8b/lora/sft"),
            "processor_class": AutoProcessor,
            "model_class": LlavaForConditionalGeneration
        },
        "llava-next": {
            "base_path": "llava-v1.6-mistral-7b-hf",
            "lora_path": os.path.join(PROJECT_ROOT, "LLaMA-Factory/saves/llava-next-7b/lora/sft"),
            "processor_class": LlavaNextProcessor,
            "model_class": LlavaNextForConditionalGeneration
        }
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def clean_memory():
    """Force clear CUDA memory to reduce OOM risk."""
    gc.collect()
    torch.cuda.empty_cache()

def clean_model_output(output_text, model_type):
    """Strip prompt echoes based on model type."""
    text = output_text.strip()
    if model_type == "llava-next" and "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    elif model_type == "llava-1.5" and "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text

def parse_ground_truth(answer_text, q_type):
    text = str(answer_text).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if q_type == "Multiple_Choice":
        text = text.upper()
        match = re.match(r"^([A-D])", text)
        return match.group(1) if match else text
    elif q_type == "True_False":
        text_lower = text.lower()
        if "true" in text_lower or "yes" in text_lower: return "True"
        if "false" in text_lower or "no" in text_lower: return "False"
        return text
    return text

def extract_answer(text, q_type):
    text = text.strip()
    if q_type == "True_False":
        clean = re.sub(r'[^\w\s]', '', text).lower()
        if "true" in clean and "false" not in clean: return "True"
        if "false" in clean and "true" not in clean: return "False"
        if re.search(r'\b(yes)\b', clean): return "True"
        if re.search(r'\b(no)\b', clean): return "False"
        return text 
    elif q_type == "Multiple_Choice":
        match = re.search(r"(?:Answer|Option|Choice)(?:\s*:|\s+is)?\s*([A-D])\b", text, re.IGNORECASE)
        if match: return match.group(1).upper()
        match = re.search(r"^([A-D])[\.\)]", text)
        if match: return match.group(1)
        match = re.search(r"\\boxed\{([A-D])\}", text)
        if match: return match.group(1)
        matches = re.findall(r"\b([A-D])\b", text)
        if matches: return matches[-1]
        return text
    return text



def run_evaluation_task(model_type, dataset_type, use_lora):
    """Run one evaluation task."""
    
    
    config = GLOBAL_CONFIG["MODELS"][model_type]
    json_filename = GLOBAL_CONFIG["DATASETS"][dataset_type]
    dataset_path = os.path.join(GLOBAL_CONFIG["BASE_DIR"][dataset_type], json_filename)
    
    output_filename = f"result_{model_type}_{dataset_type}_{'lora' if use_lora else 'base'}.json"
    output_path = os.path.join(GLOBAL_CONFIG["BASE_DIR"][dataset_type], output_filename)

    print(f"\n{'#'*60}")
    print(f"🚀 Running task: [{model_type.upper()}] | Dataset: [{dataset_type}] | LoRA: [{use_lora}]")
    print(f"📄 Reading dataset: {json_filename}")
    print(f"💾 Output file: {output_filename}")
    print(f"{'#'*60}\n")

    if not os.path.exists(dataset_path):
        print(f"❌ Error: dataset file not found {dataset_path}, skipping this task.")
        return

    
    try:
        clean_memory() 
        
        processor = config["processor_class"].from_pretrained(config["base_path"])
        base_model = config["model_class"].from_pretrained(
            config["base_path"], 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        if use_lora:
            if os.path.exists(config["lora_path"]):
                print(f"🔄 Loading LoRA adapter: {config['lora_path']}")
                model = PeftModel.from_pretrained(base_model, config["lora_path"])
                print("⚡ Merging weights (Merge and Unload)...")
                model = model.merge_and_unload()
            else:
                print(f"⚠️ Warning: LoRA path does not exist ({config['lora_path']}), fallback to Base model!")
                model = base_model
        else:
            model = base_model

        model.eval()
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    results = []
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    acc_correct = 0
    acc_total = 0

    
    eos_token_id = processor.tokenizer.eos_token_id

    for entry in tqdm(dataset, desc=f"Evaluating {model_type}-{dataset_type}"):
        image_path = os.path.join(GLOBAL_CONFIG["BASE_DIR"][dataset_type], entry['image_path'])
        q_type = entry.get('type', 'Multiple_Choice')
        is_evaluable = q_type in ["Multiple_Choice", "True_False"]

        
        if not os.path.exists(image_path):
            entry['model_output'] = "IMG_MISSING"
            results.append(entry)
            continue
        
        
        if q_type == "Multiple_Choice":
            options = "\n".join(entry['options'])
            txt = f"{entry['question']}\n{options}\nDirect answer the option number."
        elif q_type == "True_False":
            txt = f"{entry['question']}\nAnswer True or False directly."
        else:
            txt = f"{entry['question']}\nPlease output the answer directly."

        conversation = [{"role": "user", "content": [{"type": "text", "text": txt}, {"type": "image"}]}]
        
        try:
            raw_image = Image.open(image_path).convert('RGB')
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
            
            if model.dtype == torch.float16:
                inputs = inputs.to(dtype=torch.float16)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=False,
                    eos_token_id=eos_token_id
                )
            
            full_text = processor.decode(outputs[0], skip_special_tokens=True)
            cleaned_text = clean_model_output(full_text, model_type)
            
        except Exception as e:
            print(f"Inference error: {e}")
            cleaned_text = "ERROR"

        
        gt = parse_ground_truth(entry['answer'], q_type)
        pred = extract_answer(cleaned_text, q_type)
        
        is_correct = None
        if is_evaluable:
            is_correct = (gt == pred)
            if is_correct:
                acc_correct += 1
                stats[q_type]['correct'] += 1
            acc_total += 1
            stats[q_type]['total'] += 1
        
        entry.update({
            "model_output": cleaned_text,
            "extracted_answer": pred,
            "parsed_ground_truth": gt,
            "is_correct": is_correct
        })
        results.append(entry)

    
    accuracy = (acc_correct / acc_total * 100) if acc_total > 0 else 0
    print(f"✅ Task completed. Accuracy: {accuracy:.2f}%")
    
    final_output = {
        "meta": {"model": model_type, "dataset": dataset_type, "lora": use_lora, "accuracy": accuracy},
        "details": results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    
    del model
    del processor
    try:
        del base_model
    except:
        pass
    clean_memory()
    print("🧹 CUDA memory cleared.\n")



if __name__ == "__main__":
    
    
    
    models = ["llava-1.5", "llava-next"]
    datasets = ["real","sim"]
    lora_states = [True, False] # True=LoRA, False=Base

    print("🏁 Starting batch evaluation tasks...")

    for model_type in models:
        for dataset_type in datasets:
            for use_lora in lora_states:
                run_evaluation_task(model_type, dataset_type, use_lora)

    print(f"{'='*60}")
    print("🎉 All evaluation tasks finished!")
    print(f"{'='*60}")
