from typing import Dict, List, Optional
import torch
from .model_interface import UnslothModelWrapper, TGIModelWrapper, VLLMModelWrapper

def init_model(args, for_inference: bool, use_sql_model: bool = False, model_name: str = None):
    """Initialize model and tokenizer with given configuration
    
    Args:
        args: Arguments containing model configuration
        use_sql_model: If True, use SQL-specific model name and chat template
    """
    if use_sql_model:
        model_name = args.model_sql_name 
    elif model_name:
        model_name = model_name
    else:
        model_name = args.model_name
    
    if getattr(args, "backend", "unsloth") == "unsloth":
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=args.max_seq_length,
            dtype=dtype,
            load_in_4bit=args.load_in_4bit,
            token=args.hf_token,
        )
        model = UnslothModelWrapper(model, tokenizer)

        chat_template = args.sql_chat_template if use_sql_model else args.chat_template

        if chat_template:
            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        else:
            model_name_lower = model_name.lower()

            if "llama" in model_name_lower:
                chat_template = "llama-3.1"
            elif "qwen" in model_name_lower:
                chat_template = "qwen-2.5"
            elif "gemma" in model_name_lower:
                chat_template = "gemma"
            else:
                raise ValueError(f"Could not automatically determine chat template for model {model_name}. Please specify --chat_template")

            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

        if for_inference:
            FastLanguageModel.for_inference(model.model)

    elif getattr(args, "backend") == "tgi":
        from openai import OpenAI

        client = OpenAI(
            base_url=args.tgi_url,
            api_key="-"  # not used for local TGI
        )

        model = TGIModelWrapper(client)
        # For TGI, we don't need tokenizer as we'll work with raw text
        tokenizer = None

    elif getattr(args, "backend") == "vllm":
        from openai import OpenAI

        client = OpenAI(
            base_url=args.vllm_url,
            api_key="-"  # not used for local vLLM
        )

        model = VLLMModelWrapper(client, model_name=model_name)
        # For vLLM, we don't need tokenizer as we'll work with raw text
        tokenizer = None

    else:
        raise ValueError(f"Unknown backend: {getattr(args, 'backend', 'unsloth')}")
    
    return model, tokenizer

def get_generation_config(args, model) -> Dict:
    """
    Get generation configuration, using model defaults and overriding with any specified command line arguments
    """
    if isinstance(model, UnslothModelWrapper):
        # Start with model's default generation config for Unsloth
        config = model.model.generation_config.to_dict()

        # Override with any specified command line arguments
        for param in ['max_new_tokens', 'do_sample', 'num_beams', 'temperature', 'top_p', 'top_k']:
            if hasattr(args, param) and getattr(args, param) is not None:
                config[param] = getattr(args, param)

        # Always set use_cache=True as it's a performance optimization
        config['use_cache'] = True

    elif isinstance(model, (TGIModelWrapper, VLLMModelWrapper)):
        # For TGI and vLLM, only include parameters that were explicitly specified
        config = {}
        for param, api_param in [
            ('temperature', 'temperature'),
            ('top_p', 'top_p'),
            ('max_new_tokens', 'max_tokens')
        ]:
            if hasattr(args, param) and getattr(args, param) is not None:
                config[api_param] = getattr(args, param)

    return config

def generate_from_prompt(
    model_wrapper,
    tokenizer,
    messages: List[Dict[str, str]],
    generation_config: Dict,
    max_length: Optional[int] = None
) -> Optional[str]:
    """
    Generate text using the model with given messages and configuration

    Args:
        model: The language model (UnslothModelWrapper, TGIModelWrapper, or VLLMModelWrapper)
        tokenizer: The tokenizer (None for TGI and vLLM)
        messages: List of message dictionaries with 'role' and 'content'
        generation_config: Generation configuration dictionary
        max_length: Optional maximum length check

    Returns:
        Generated text or None if input is too long
    """
    if isinstance(model_wrapper, UnslothModelWrapper):
        # Unsloth path - use tokenizer and model.generate
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True
        )

        if max_length and len(inputs[0]) >= max_length:
            print("Generation input is too long")
            return None

        output_tokens = model_wrapper.generate(
            inputs,
            **generation_config
        )

        return tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True)

    elif isinstance(model_wrapper, (TGIModelWrapper, VLLMModelWrapper)):
        # TGI and vLLM path - use OpenAI-compatible API
        return model_wrapper.generate(messages, **generation_config)
