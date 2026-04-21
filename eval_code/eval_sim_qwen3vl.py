import json
import os
import torch
import re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from collections import defaultdict
from peft import PeftModel  

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = "Qwen3-VL-32B-Instruct"
LORA_PATH = os.path.join(PROJECT_ROOT, "LLaMA-Factory/saves/qwen3vl-32b/lora/sft")
IS_LORA = True

BASE_DIR = os.path.join(PROJECT_ROOT, "simdata_benchmark")
JSON_FILE = "pure_sim_test.json"

OUTPUT_FILE = "sim_testset_qwen3_32b_lora.json" if IS_LORA else "sim_testset_qwen3_32b.json"

print(f"Loading model: {MODEL_PATH} ...")

try:
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    if os.path.exists(LORA_PATH) and IS_LORA:
        print(f"Loading LoRA weights: {LORA_PATH} ...")
        
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        
        
        
        
        print("Merging LoRA weights into base model (merge_and_unload)...")
        model = model.merge_and_unload()
    else:
        print(f"Warning: LoRA path not found {LORA_PATH}, inference will use base model only!")
        model = base_model

    model.eval()
    print("Model loaded.")
except Exception as e:
    print(f"Model loading failed: {e}")
    exit(1)

def parse_ground_truth(answer_text, q_type):
    """
    Parse the ground-truth answer.
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
            return "True"
        return text

    
    elif q_type == "Open_Ended":
        return text

    return text

def extract_answer(text, q_type):
    """
    Extract the answer from model output.
    """
    text = text.strip()
    
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
        
        
        match = re.search(r"(?:Answer|Option|Choice)(?:\s*:|\s+is)?\s*([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        
        
        match = re.search(r"^([A-D])[\.\)]", text)
        if match:
            return match.group(1)

        
        match = re.search(r"\\boxed\{([A-D])\}", text)
        if match:
            return match.group(1)

        
        
        
        matches = re.findall(r"\b([A-D])\b", text)
        if matches:
            
            return matches[-1]
            
        return text

    
    
    elif q_type == "Open_Ended":
        return text
    
    return text

def main():
    json_path = os.path.join(BASE_DIR, JSON_FILE)
    
    if not os.path.exists(json_path):
        print(f"Error: dataset file not found {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples. Starting inference...")
    print("Note: Open_Ended is generation-only and excluded from accuracy.")
    
    results = []

    
    
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'is_evaluable': False})
    dimension_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    deixis_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    
    acc_correct_count = 0
    acc_total_count = 0

    for entry in tqdm(dataset):
        image_rel_path = entry['image_path']
        question = entry['question']
        raw_answer = entry['answer']
        q_type = entry.get('type', 'Multiple_Choice') 
        
        dimension = entry.get('dimension', 'Unknown_Dimension')
        deixis_level = entry.get('deixis_level', 'Unknown_Level')
        
        
        is_evaluable_type = q_type in ["Multiple_Choice", "True_False"]
        type_stats[q_type]['is_evaluable'] = is_evaluable_type

        
        image_full_path = os.path.join(BASE_DIR, image_rel_path)
        if not os.path.exists(image_full_path):
            entry['model_output'] = "IMAGE_NOT_FOUND"
            entry['extracted_answer'] = "None"
            entry['is_correct'] = False if is_evaluable_type else None
            results.append(entry)
            
            
            type_stats[q_type]['total'] += 1
            if is_evaluable_type:
                dimension_stats[dimension]['total'] += 1
                deixis_stats[deixis_level]['total'] += 1
                acc_total_count += 1
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

        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            
            max_tokens = 128 if q_type == "Open_Ended" else 16
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

        
        ground_truth = parse_ground_truth(raw_answer, q_type)
        extracted_ans = extract_answer(output_text, q_type)
        
        if is_evaluable_type:
            
            is_correct = (extracted_ans == ground_truth)
            
            if is_correct:
                acc_correct_count += 1
                type_stats[q_type]['correct'] += 1
                dimension_stats[dimension]['correct'] += 1
                deixis_stats[deixis_level]['correct'] += 1
            
            
            acc_total_count += 1
            dimension_stats[dimension]['total'] += 1
            deixis_stats[deixis_level]['total'] += 1
        else:
            
            is_correct = None 

        type_stats[q_type]['total'] += 1

        
        entry['model_output'] = output_text
        entry['extracted_answer'] = extracted_ans
        entry['parsed_ground_truth'] = ground_truth
        entry['is_correct'] = is_correct 
        results.append(entry)

    
    
    
    total_accuracy = (acc_correct_count / acc_total_count) * 100 if acc_total_count > 0 else 0
    
    print("\n" + "="*60)
    print(f"[Evaluation Summary]")
    print(f"Total processed: {len(dataset)}")
    print(f"  - Auto-evaluated (MC+TF): {acc_total_count}")
    print(f"  - Generation-only (Open_Ended): {type_stats['Open_Ended']['total']}")
    print("-" * 60)
    print(f"[Accuracy (MC + TF only)]")
    print(f"Correct: {acc_correct_count}")
    print(f"Overall accuracy: {total_accuracy:.2f}%")
    print("="*60)

    
    print(f"\n[Breakdown by Type]")
    print(f"{'Type':<20} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 55)
    for q_type, stats in sorted(type_stats.items()):
        if stats['is_evaluable']:
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            acc_str = f"{acc:.2f}%"
        else:
            acc_str = "N/A" 
        print(f"{q_type:<20} | {stats['total']:<8} | {stats['correct']:<8} | {acc_str:<8}")

    
    print(f"\n[Breakdown by Dimension (MC/TF only)]")
    print(f"{'Dimension':<30} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 65)
    for dim, stats in sorted(dimension_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{dim:<30} | {stats['total']:<8} | {stats['correct']:<8} | {acc:.2f}%")

    
    print(f"\n[Breakdown by Deixis Level (MC/TF only)]")
    print(f"{'Deixis Level':<20} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 55)
    for level, stats in sorted(deixis_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{level:<20} | {stats['total']:<8} | {stats['correct']:<8} | {acc:.2f}%")
    
    print("="*60)

    
    output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
