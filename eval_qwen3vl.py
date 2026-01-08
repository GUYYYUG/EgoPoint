import json
import os
import torch
import re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig 
from collections import defaultdict
from peft import PeftModel

# ================= Configuration =================
# Model Paths (Using relative paths)
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"  # you can change to anthor files
LORA_PATH = "saves/model/lora/sft"
IS_LORA = False  # Set to True if you have LoRA weights at the path above

# Dataset Configuration
JSON_FILE = "mini_dataset.json" 
IMAGE_DIR = "mini_dataset_img"
OUTPUT_FILE = "evaluation_qwen_results.json"

# ================= Load Model =================
print(f"Loading model from: {MODEL_PATH} ...")

try:
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 2. Load LoRA Adapters
    if os.path.exists(LORA_PATH) and IS_LORA:
        print(f"Loading LoRA weights: {LORA_PATH} ...")
        # Mount LoRA to base model
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        
        # 3. Merge Weights (Merge and Unload)
        # This permanently adds LoRA weights to base weights and removes Peft wrapper.
        # Faster inference speed.
        print("Merging LoRA weights into base model (merge_and_unload)...")
        model = model.merge_and_unload()
    else:
        if IS_LORA:
            print(f"Warning: LoRA path {LORA_PATH} not found. Using base model only!")
        else:
            print("Using base model (LoRA disabled).")
        model = base_model

    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    # In a real script you might want to exit, but we'll print error
    # exit(1) 

# ================= Helper Functions =================

def parse_ground_truth(answer_text, q_type):
    """
    Parse the ground truth answer.
    """
    text = str(answer_text).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    if q_type == "Multiple_Choice":
        text = text.upper()
        match = re.match(r"^([A-D])", text)
        if match:
            return match.group(1)
        return text
        
    elif q_type == "True_False":
        text_lower = text.lower()
        if "true" in text_lower:
            return "True"
        if "false" in text_lower:
            return "False"
        if "yes" in text_lower:
            return "True"
        if "no" in text_lower:
            return "False"
        return text

    # Open_Ended returns original text
    elif q_type == "Open_Ended":
        return text

    return text

def extract_answer(text, q_type):
    """
    Extract answer from model output.
    """
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    if q_type == "True_False":
        clean_text = re.sub(r'[^\w\s]', '', text).lower()
        if "true" in clean_text and "false" not in clean_text:
            return "True"
        if "false" in clean_text and "true" not in clean_text:
            return "False"
        if re.search(r'\b(yes)\b', clean_text):
            return "True"
        if re.search(r'\b(no)\b', clean_text):
            return "False"
        return text 
    elif q_type == "Multiple_Choice":
        # 1. Prioritize explicit "Answer: X" or "Option: X" format
        match = re.search(r"(?:Answer|Option|Choice)(?:\s*:|\s+is)?\s*([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # 2. Match option at start of sentence (e.g., "B. On a shelf")
        match = re.search(r"^([A-D])[\.\)]", text)
        if match:
            return match.group(1)

        # 3. Match LaTeX format
        match = re.search(r"\\boxed\{([A-D])\}", text)
        if match:
            return match.group(1)

        # 4. Fallback strategy
        matches = re.findall(r"\b([A-D])\b", text)
        if matches:
            # Only take the last capital letter if no other features found
            return matches[-1]
            
        return text

    # Open_Ended does not use regex extraction
    elif q_type == "Open_Ended":
        return text
    
    return text

# ================= Main Logic =================

def main():
    if not os.path.exists(JSON_FILE):
        print(f"Error: Dataset file not found at {JSON_FILE}")
        return

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} items. Starting inference...")
    print("Note: Open_Ended type only generates results and is not counted in accuracy statistics.")
    
    results = []

    # Statistics Dictionaries
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'is_evaluable': False})
    dimension_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    deixis_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Accuracy counters (MC and TF only)
    acc_correct_count = 0
    acc_total_count = 0

    for entry in tqdm(dataset):
        # ADAPTATION: Using 'image_id' to construct path instead of 'image_path'
        image_id = entry.get('image_id')
        question = entry.get('question', '')
        raw_answer = entry.get('answer', '')
        q_type = entry.get('type', 'Multiple_Choice') 
        
        dimension = entry.get('dimension', 'Unknown_Dimension')
        deixis_level = entry.get('deixis_level', 'Unknown_Level')
        
        # 1. Check if type is evaluable
        is_evaluable_type = q_type in ["Multiple_Choice", "True_False"]
        type_stats[q_type]['is_evaluable'] = is_evaluable_type

        # 2. Image Check
        # ADAPTATION: Construct path using IMAGE_DIR and image_id
        if image_id:
            image_full_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        else:
            image_full_path = "INVALID_PATH"

        if not os.path.exists(image_full_path):
            entry['model_output'] = "IMAGE_NOT_FOUND"
            entry['extracted_answer'] = "None"
            entry['is_correct'] = False if is_evaluable_type else None
            results.append(entry)
            
            # Count only total
            type_stats[q_type]['total'] += 1
            if is_evaluable_type:
                dimension_stats[dimension]['total'] += 1
                deixis_stats[deixis_level]['total'] += 1
                acc_total_count += 1
            continue

        # 3. Construct Prompt
        if q_type == "Multiple_Choice":
            options = entry.get('options', [])
            if isinstance(options, list):
                options_str = "\n".join(options)
            else:
                options_str = str(options)
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

        # 4. Inference
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                max_tokens = 128
                generated_ids = model.generate(**inputs, max_new_tokens=max_tokens,
                                             do_sample=False,
                                             temperature=0.0,
                                             top_p=1.0)
        
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except Exception as e:
            output_text = f"ERROR: {str(e)}"

        # 5. Post-processing and Evaluation
        ground_truth = parse_ground_truth(raw_answer, q_type)
        extracted_ans = extract_answer(output_text, q_type)
        
        if is_evaluable_type:
            # Automatic scoring for MC and TF
            is_correct = (extracted_ans == ground_truth)
            
            if is_correct:
                acc_correct_count += 1
                type_stats[q_type]['correct'] += 1
                dimension_stats[dimension]['correct'] += 1
                deixis_stats[deixis_level]['correct'] += 1
            
            # Update totals
            acc_total_count += 1
            dimension_stats[dimension]['total'] += 1
            deixis_stats[deixis_level]['total'] += 1
        else:
            # Open_Ended
            is_correct = None 

        type_stats[q_type]['total'] += 1

        # Record Results
        entry['model_output'] = output_text
        entry['extracted_answer'] = extracted_ans
        entry['parsed_ground_truth'] = ground_truth
        entry['is_correct'] = is_correct 
        results.append(entry)

    # ================= Results Statistics & Display =================
    
    # Calculate Total Accuracy (MC + TF only)
    total_accuracy = (acc_correct_count / acc_total_count) * 100 if acc_total_count > 0 else 0
    
    print("\n" + "="*60)
    print(f"【Evaluation Summary】")
    print(f"Total Processed: {len(dataset)}")
    print(f"  - Auto-evaluable (MC+TF): {acc_total_count}")
    print(f"  - Generation Only (Open_Ended): {type_stats['Open_Ended']['total']}")
    print("-" * 60)
    print(f"【Accuracy Statistics (MC + TF Only)】")
    print(f"Correct Count: {acc_correct_count}")
    print(f"Overall Accuracy: {total_accuracy:.2f}%")
    print("="*60)

    # 1. Analyze by Type
    print(f"\n【Analysis by Type】")
    print(f"{'Type':<20} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 55)
    for q_type, stats in sorted(type_stats.items()):
        if stats['is_evaluable']:
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            acc_str = f"{acc:.2f}%"
        else:
            acc_str = "N/A"
        print(f"{q_type:<20} | {stats['total']:<8} | {stats['correct']:<8} | {acc_str:<8}")

    # 2. Analyze by Dimension
    print(f"\n【Analysis by Dimension (MC/TF Only)】")
    print(f"{'Dimension':<30} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 65)
    for dim, stats in sorted(dimension_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{dim:<30} | {stats['total']:<8} | {stats['correct']:<8} | {acc:.2f}%")

    # 3. Analyze by Deixis Level
    print(f"\n【Analysis by Deixis Level (MC/TF Only)】")
    print(f"{'Deixis Level':<20} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 55)
    for level, stats in sorted(deixis_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{level:<20} | {stats['total']:<8} | {stats['correct']:<8} | {acc:.2f}%")
    
    print("="*60)

    # Save Results
    final_output = {
        "summary": {
            "total_processed": len(dataset),
            "mc_tf_accuracy": total_accuracy,
            "counts": {
                "mc_tf_total": acc_total_count,
                "open_ended_total": type_stats['Open_Ended']['total']
            },
            "type_breakdown": dict(type_stats),
            "dimension_breakdown_mc_tf": dict(dimension_stats), 
        },
        "details": results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"Detailed results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
