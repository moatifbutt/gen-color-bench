"""
VLM (Vision-Language Model) loading and inference.

Wraps Janus VLM for object presence verification.
"""

import os
import torch
from dataclasses import dataclass
from PIL import Image
from typing import Dict, List, Optional, Any


DEFAULT_CACHE_DIR = "/data/144-1/users/mabutt/gencolorbench/cache"


@dataclass
class VLMModels:
    """Container for VLM models."""
    model: Any  # MultiModalityCausalLM
    processor: Any  # VLChatProcessor
    tokenizer: Any
    device: str
    variant: str  # "1.3B" or "7B"


@dataclass
class VLMCheckResult:
    """Result of VLM object presence check."""
    objects_checked: Dict[str, str]  # object_name -> "yes"/"no"/"error"
    all_present: bool
    main_obj_present: Optional[bool] = None  # For Task 3
    sec_obj_present: Optional[bool] = None   # For Task 3
    skip_segmentation: bool = False
    early_decision: Optional[bool] = None    # If we can decide without segmentation
    early_decision_reason: Optional[str] = None


def load_vlm_model(
    model_path: str,
    device: str,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> VLMModels:
    """
    Load Janus VLM for object presence verification.
    
    Supported models:
    - deepseek-ai/Janus-1.3B (~3GB VRAM)
    - deepseek-ai/Janus-Pro-7B (~14GB VRAM)
    
    Args:
        model_path: HuggingFace model ID or local path
        device: Device string
        cache_dir: Cache directory for models
    
    Returns:
        VLMModels container
    """
    print("=" * 60)
    print("Loading VLM model...")
    print("=" * 60)
    
    # Setup cache environment
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cache_dir}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{cache_dir}/datasets"
    os.environ["TOKENIZERS_CACHE"] = f"{cache_dir}/tokenizers"
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(f"{cache_dir}/hub", exist_ok=True)
    os.makedirs(f"{cache_dir}/transformers", exist_ok=True)
    
    from transformers import AutoConfig
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    
    is_7b = "7B" in model_path or "7b" in model_path
    variant = "7B" if is_7b else "1.3B"
    
    print(f"Loading Janus from {model_path}...")
    print(f"Model variant: {variant}")
    print(f"Cache directory: {cache_dir}")
    
    config = AutoConfig.from_pretrained(model_path, cache_dir=f"{cache_dir}/hub")
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    
    model = MultiModalityCausalLM.from_pretrained(
        model_path,
        language_config=language_config,
        trust_remote_code=True,
        cache_dir=f"{cache_dir}/hub",
    )
    model = model.to(torch.bfloat16).to(device).eval()
    
    processor = VLChatProcessor.from_pretrained(model_path, cache_dir=f"{cache_dir}/hub")
    tokenizer = processor.tokenizer
    
    vram_estimate = "~14GB" if is_7b else "~3GB"
    print(f"✓ Janus VLM loaded (bfloat16, {vram_estimate} VRAM)")
    
    return VLMModels(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        variant=variant,
    )


def _run_vlm_inference(
    vlm: VLMModels,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 30
) -> str:
    """
    Run VLM inference with a prompt.
    
    Args:
        vlm: VLM models container
        image_path: Path to image
        prompt: Text prompt with <image_placeholder>
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Model response string
    """
    conversation = [
        {
            "role": "User",
            "content": prompt,
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    pil_image = Image.open(image_path).convert("RGB")
    
    prepare_inputs = vlm.processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
    ).to(vlm.device)
    
    inputs_embeds = vlm.model.prepare_inputs_embeds(**prepare_inputs)
    
    with torch.no_grad():
        outputs = vlm.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=vlm.tokenizer.eos_token_id,
            bos_token_id=vlm.tokenizer.bos_token_id,
            eos_token_id=vlm.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    
    answer = vlm.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer.strip().lower()


def vlm_check_single_object(
    vlm: VLMModels,
    image_path: str,
    object_name: str,
    generation_prompt: Optional[str] = None
) -> VLMCheckResult:
    """
    Check if a single object is present in the image.
    
    Args:
        vlm: VLM models container
        image_path: Path to image
        object_name: Object to look for
        generation_prompt: Original generation prompt (optional)
    
    Returns:
        VLMCheckResult with presence info
    """
    try:
        if generation_prompt:
            prompt = (
                f"<image_placeholder>\n"
                f"You are a visual object presence evaluator. An image generator was given this prompt: '{generation_prompt}'.\n\n"
                f"Critically examine this image: Is the {object_name} clearly visible as a DISTINCT, SEPARATE object?\n\n"
                f"Answer ONLY 'yes' or 'no'."
            )
        else:
            prompt = (
                f"<image_placeholder>\n"
                f"You are a visual object presence evaluator. Critically examine this image.\n\n"
                f"Question: Is there a {object_name} clearly visible as a DISTINCT, SEPARATE object?\n\n"
                f"Answer ONLY 'yes' or 'no'."
            )
        
        answer = _run_vlm_inference(vlm, image_path, prompt, max_new_tokens=10)
        
        # Parse response
        answer_clean = answer.replace('.', '').replace(',', '').strip()
        
        if answer_clean == "yes" or answer_clean.startswith("yes"):
            result = "yes"
        elif answer_clean == "no" or answer_clean.startswith("no"):
            result = "no"
        elif "yes" in answer and "no" not in answer:
            result = "yes"
        elif "no" in answer and "yes" not in answer:
            result = "no"
        else:
            result = "unknown"
        
        is_present = (result == "yes")
        
        if not is_present:
            return VLMCheckResult(
                objects_checked={object_name: result},
                all_present=False,
                main_obj_present=False,
                skip_segmentation=True,
                early_decision=False,
                early_decision_reason="object_missing"
            )
        
        return VLMCheckResult(
            objects_checked={object_name: result},
            all_present=True,
            main_obj_present=True,
            skip_segmentation=False,
        )
    
    except Exception as e:
        import traceback
        print(f"VLM error for '{object_name}': {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return VLMCheckResult(
            objects_checked={object_name: "error"},
            all_present=False,
            skip_segmentation=False,
        )


def vlm_check_two_objects(
    vlm: VLMModels,
    image_path: str,
    main_obj: str,
    sec_obj: str,
    generation_prompt: Optional[str] = None
) -> VLMCheckResult:
    """
    Check if two objects are present (Task 3 - single VLM call).
    
    Args:
        vlm: VLM models container
        image_path: Path to image
        main_obj: Main object name
        sec_obj: Secondary object name
        generation_prompt: Original generation prompt
    
    Returns:
        VLMCheckResult with presence info for both objects
    """
    try:
        if generation_prompt:
            prompt = (
                f"<image_placeholder>\n"
                f"You are a visual object presence evaluator. An image generator was given this prompt: '{generation_prompt}'.\n\n"
                f"Critically examine this image for these TWO objects:\n"
                f"1. {main_obj}\n"
                f"2. {sec_obj}\n\n"
                f"Answer in this EXACT format:\n"
                f"{main_obj}: yes/no\n"
                f"{sec_obj}: yes/no\n"
                f"BOTH separate: yes/no"
            )
        else:
            prompt = (
                f"<image_placeholder>\n"
                f"You are a visual object presence evaluator. Critically examine this image.\n\n"
                f"Check for these TWO objects:\n"
                f"1. {main_obj}\n"
                f"2. {sec_obj}\n\n"
                f"Answer in this EXACT format:\n"
                f"{main_obj}: yes/no\n"
                f"{sec_obj}: yes/no\n"
                f"BOTH separate: yes/no"
            )
        
        answer = _run_vlm_inference(vlm, image_path, prompt, max_new_tokens=30)
        
        # Parse response
        results = {'main_obj': 'unknown', 'sec_obj': 'unknown', 'both': 'unknown'}
        
        lines = answer.split('\n')
        for line in lines:
            line = line.strip().lower()
            
            if main_obj.lower() in line:
                if 'yes' in line and 'no' not in line.split('yes')[0]:
                    results['main_obj'] = 'yes'
                elif 'no' in line:
                    results['main_obj'] = 'no'
            
            if sec_obj.lower() in line:
                if 'yes' in line and 'no' not in line.split('yes')[0]:
                    results['sec_obj'] = 'yes'
                elif 'no' in line:
                    results['sec_obj'] = 'no'
            
            if 'both' in line:
                if 'yes' in line and 'no' not in line.split('yes')[0]:
                    results['both'] = 'yes'
                elif 'no' in line:
                    results['both'] = 'no'
        
        main_present = (results['main_obj'] == 'yes')
        sec_present = (results['sec_obj'] == 'yes')
        
        # If both individually say yes but combined says no, trust combined
        if main_present and sec_present and results['both'] == 'no':
            sec_present = False
        
        objects_checked = {
            main_obj: results['main_obj'],
            sec_obj: results['sec_obj'],
            'both_verified': results['both']
        }
        
        if not main_present:
            return VLMCheckResult(
                objects_checked=objects_checked,
                all_present=False,
                main_obj_present=False,
                sec_obj_present=sec_present,
                skip_segmentation=True,
                early_decision=False,
                early_decision_reason="main_obj_missing"
            )
        
        return VLMCheckResult(
            objects_checked=objects_checked,
            all_present=main_present and sec_present,
            main_obj_present=True,
            sec_obj_present=sec_present,
            skip_segmentation=False,
        )
    
    except Exception as e:
        import traceback
        print(f"VLM error for two-object check: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return VLMCheckResult(
            objects_checked={main_obj: 'error', sec_obj: 'error'},
            all_present=False,
            skip_segmentation=False,
        )


def vlm_check_multi_objects(
    vlm: VLMModels,
    image_path: str,
    objects: List[str],
    generation_prompt: Optional[str] = None
) -> VLMCheckResult:
    """
    Check if multiple objects are present (Task 4 - single VLM call).
    
    Args:
        vlm: VLM models container
        image_path: Path to image
        objects: List of object names
        generation_prompt: Original generation prompt
    
    Returns:
        VLMCheckResult with presence info for all objects
    """
    try:
        obj_list = "\n".join([f"- {obj}" for obj in objects])
        
        if generation_prompt:
            prompt = (
                f"<image_placeholder>\n"
                f"You are a visual object presence evaluator. An image generator was given this prompt: '{generation_prompt}'.\n\n"
                f"Critically examine this image and check if each of these objects is present as a DISTINCT, SEPARATE object:\n"
                f"{obj_list}\n\n"
                f"For EACH object, answer 'yes' if clearly present, or 'no' if missing.\n"
                f"Format your response EXACTLY as:\n"
                f"{chr(10).join([f'{obj}: yes/no' for obj in objects])}"
            )
        else:
            prompt = (
                f"<image_placeholder>\n"
                f"You are a visual object presence evaluator. Critically examine this image.\n\n"
                f"Check if each of these objects is present as a DISTINCT, SEPARATE object:\n"
                f"{obj_list}\n\n"
                f"For EACH object, answer 'yes' if clearly present, or 'no' if missing.\n"
                f"Format your response EXACTLY as:\n"
                f"{chr(10).join([f'{obj}: yes/no' for obj in objects])}"
            )
        
        answer = _run_vlm_inference(vlm, image_path, prompt, max_new_tokens=50)
        
        # Parse response
        results = {}
        for obj in objects:
            obj_lower = obj.lower()
            found = False
            
            for line in answer.split('\n'):
                line = line.strip().lower()
                if obj_lower in line:
                    if 'yes' in line and 'no' not in line.split('yes')[0]:
                        results[obj] = "yes"
                        found = True
                        break
                    elif 'no' in line:
                        results[obj] = "no"
                        found = True
                        break
            
            if not found:
                if obj_lower in answer:
                    idx = answer.find(obj_lower)
                    context = answer[max(0, idx-20):min(len(answer), idx+len(obj_lower)+20)]
                    if 'yes' in context and 'no' not in context:
                        results[obj] = "yes"
                    elif 'no' in context:
                        results[obj] = "no"
                    else:
                        results[obj] = "unknown"
                else:
                    results[obj] = "unknown"
        
        all_present = True
        missing_objects = []
        
        for obj in objects:
            if results.get(obj) != "yes":
                all_present = False
                missing_objects.append(obj)
        
        if not all_present:
            return VLMCheckResult(
                objects_checked=results,
                all_present=False,
                skip_segmentation=True,
                early_decision=False,
                early_decision_reason=f"objects_missing: {', '.join(missing_objects)}"
            )
        
        return VLMCheckResult(
            objects_checked=results,
            all_present=True,
            skip_segmentation=False,
        )
    
    except Exception as e:
        import traceback
        print(f"VLM error for multi-object check: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return VLMCheckResult(
            objects_checked={obj: "error" for obj in objects},
            all_present=False,
            skip_segmentation=False,
        )
