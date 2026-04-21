import json
import os
import torch
import re
import gc 
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))



MODEL_CONFIGS = {
    "2b": {
        "base": "InternVL3_5-2B-HF",  
        "lora": os.path.join(PROJECT_ROOT, "LLaMA-Factory/saves/intern_vl-2b/lora/sft")  
    },
    "8b": {
        "base": "InternVL3_5-8B-HF",       
        "lora": os.path.join(PROJECT_ROOT, "LLaMA-Factory/saves/intern_vl-8b/lora/sft")  
    },
    "14b": {
        "base": "InternVL3_5-14B-HF",      
        "lora": os.path.join(PROJECT_ROOT, "LLaMA-Factory/saves/intern_vl-14b/lora/sft") 
    }
}



def parse_ground_truth(answer_text, q_type):
    """Parse the ground-truth answer."""
    text = str(answer_text).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    if q_type == "Multiple_Choice":
        text = text.upper()
        match = re.match(r"^([A-D])", text)
        if match: return match.group(1)
        return text
    elif q_type == "True_False":
        text_lower = text.lower()
        if "true" in text_lower or "yes" in text_lower: return "True"
        if "false" in text_lower or "no" in text_lower: return "False"
        return text
    elif q_type == "Open_Ended":
        return text
    return text

def extract_answer(text, q_type):
    """Extract the answer from model output."""
    text = text.strip()
    if q_type == "True_False":
        clean_text = re.sub(r'[^\w\s]', '', text).lower()
        if "true" in clean_text and "false" not in clean_text: return "True"
        if "false" in clean_text and "true" not in clean_text: return "False"
        if re.search(r'\b(yes)\b', clean_text): return "True"
        if re.search(r'\b(no)\b', clean_text): return "False"
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
    elif q_type == "Open_Ended":
        return text
    return text



def evaluate_model(model_size_key, use_lora, test_type):
    """
    Args:
    - model_size_key: "2b", "8b", "14b"
    - use_lora: True/False
    - test_type: "real", "sim"
    """
    print(f"\n{'#'*60}")
    print(f"Start task: Model={model_size_key.upper()} | LoRA={use_lora} | Data={test_type}")
    print(f"{'#'*60}\n")

    
    config = MODEL_CONFIGS.get(model_size_key)
    if not config:
        print(f"Error: key not found in MODEL_CONFIGS '{model_size_key}'")
        return

    model_path = config["base"]
    lora_path = config["lora"]
    
    
    if test_type == "real":
        base_dir = os.path.join(PROJECT_ROOT, "realdata_benchmark")
        json_file = "pure_real_test.json"
    else:
        base_dir = os.path.join(PROJECT_ROOT, "simdata_benchmark")
        json_file = "pure_sim_test.json"
    
    
    model_suffix = f"internvl_{model_size_key}"
    output_filename = f"{test_type}_testset_{model_suffix}_lora.json" if use_lora else f"{test_type}_testset_{model_suffix}.json"
    
    print(f"-> Base model path: {model_path}")
    print(f"-> LoRA path: {lora_path if use_lora else 'Not used'}")
    print(f"-> Dataset: {os.path.join(base_dir, json_file)}")
    print(f"-> Output file: {os.path.join(base_dir, output_filename)}")

    
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print("Loading base model...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

        if use_lora:
            if os.path.exists(lora_path):
                print(f"Loading and merging LoRA: {lora_path}")
                model = PeftModel.from_pretrained(base_model, lora_path)
                model = model.merge_and_unload()
            else:
                print(f"Warning: LoRA path not found {lora_path}, fallback to base model only!")
                model = base_model
        else:
            model = base_model

        model.eval()
        eos_token_id = processor.tokenizer.eos_token_id

    except Exception as e:
        print(f"Fatal error: model load failed: {e}")
        return 

    
    json_path = os.path.join(base_dir, json_file)
    if not os.path.exists(json_path):
        print(f"Error: dataset file not found {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    
    results = []
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'is_evaluable': False})
    acc_correct_count = 0
    acc_total_count = 0

    print(f"Start inference on {len(dataset)} samples...")
    
    for entry in tqdm(dataset, desc=f"Eval {model_size_key}-{test_type}"):
        image_rel_path = entry['image_path']
        question = entry['question']
        raw_answer = entry['answer']
        q_type = entry.get('type', 'Multiple_Choice') 
        is_evaluable_type = q_type in ["Multiple_Choice", "True_False"]
        type_stats[q_type]['is_evaluable'] = is_evaluable_type

        
        image_full_path = os.path.join(base_dir, image_rel_path)
        if not os.path.exists(image_full_path):
            entry.update({'model_output': "IMAGE_NOT_FOUND", 'extracted_answer': "None", 'is_correct': False if is_evaluable_type else None})
            results.append(entry)
            type_stats[q_type]['total'] += 1
            if is_evaluable_type: acc_total_count += 1
            continue

        
        if q_type == "Multiple_Choice":
            options_str = "\n".join(entry['options'])
            prompt_text = f"{question}\n{options_str}\nAnswer directly using the letters of the options given."
        elif q_type == "True_False":
            prompt_text = f"{question}\nAnswer directly with 'True' or 'False'."
        elif q_type == "Open_Ended":
            prompt_text = f"{question}\nPlease output the answer directly."
        else:
            prompt_text = f"{question}"

        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_full_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)

            max_new_tokens = 128 if q_type == "Open_Ended" else 20
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, num_beams=1, pad_token_id=eos_token_id
                )
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"Single-sample inference failed: {e}")
            output_text = "ERROR"

        
        ground_truth = parse_ground_truth(raw_answer, q_type)
        extracted_ans = extract_answer(output_text, q_type)
        
        is_correct = None
        if is_evaluable_type:
            is_correct = (extracted_ans == ground_truth)
            if is_correct:
                acc_correct_count += 1
                type_stats[q_type]['correct'] += 1
            acc_total_count += 1
        
        type_stats[q_type]['total'] += 1
        
        entry.update({
            'model_output': output_text,
            'extracted_answer': extracted_ans,
            'parsed_ground_truth': ground_truth,
            'is_correct': is_correct
        })
        results.append(entry)

    
    total_accuracy = (acc_correct_count / acc_total_count) * 100 if acc_total_count > 0 else 0
    print(f"Task complete. Overall accuracy (MC+TF): {total_accuracy:.2f}%")
    
    output_path = os.path.join(base_dir, output_filename)
    final_output = {
        "summary": {
            "model": f"InternVL-{model_size_key.upper()}-{'LoRA' if use_lora else 'Base'}",
            "mc_tf_accuracy": total_accuracy,
            "counts": {"mc_tf_total": acc_total_count, "open_ended_total": type_stats['Open_Ended']['total']},
            "type_breakdown": dict(type_stats)
        },
        "details": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    
    print("Clearing CUDA memory...")
    del model
    del processor
    if 'inputs' in locals(): del inputs
    if 'base_model' in locals(): del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA memory cleared.\n")



if __name__ == "__main__":
    
    
    
    tasks = [
        
        ("2b", True, "real"),
        ("2b", False, "real"),
        ("2b", True, "sim"),
        ("2b", False, "sim"),
        
        
        ("8b", True, "real"),
        ("8b", False, "real"),
        ("8b", True, "sim"),
        ("8b", False, "sim"),
        
        
        ("14b", True, "real"),
        ("14b", False, "real"),
        ("14b", True, "sim"),
        ("14b", False, "sim"),
        
    ]

    print(f"Planned tasks: {len(tasks)}")
    
    for size, lora, dtype in tasks:
        evaluate_model(size, lora, dtype)
        
    print("All tasks completed!")
