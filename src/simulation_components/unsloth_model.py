"""Language Model that uses Unsloth for local inference."""

from collections.abc import Collection, Sequence
from typing import Any, Mapping
import torch

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    UNSLOTH_IMPORT_ERROR: ImportError | None = None

except ImportError as e:
    FastLanguageModel = None
    UNSLOTH_AVAILABLE = False
    UNSLOTH_IMPORT_ERROR = e

from transformers import StoppingCriteria, StoppingCriteriaList

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_TERMINATORS = ()
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_TOKENS = 5000

class StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_string, prompt_len):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only look at the tokens generated AFTER the prompt
        generated_tokens = input_ids[0, self.prompt_len:]
        if len(generated_tokens) < 1:
            return False
            
        # Check if the stop string is in the last 15 tokens of the NEWLY generated text
        decoded_gen = self.tokenizer.decode(generated_tokens[-15:])
        return self.stop_string in decoded_gen

class UnslothLanguageModel:
  """Language model wrapper for Unsloth local inference."""

  def __init__(
      self,
      model_name: str,
      *,
      max_seq_length: int = 4096,
      load_in_4bit: bool = False,
      **kwargs: Any,
  ):
    if not UNSLOTH_AVAILABLE:
        if UNSLOTH_IMPORT_ERROR is not None:
            raise ImportError(
                "Unsloth is required but failed to import. Original error: "
                + str(UNSLOTH_IMPORT_ERROR)
            ) from UNSLOTH_IMPORT_ERROR
        raise ImportError(
            "Unsloth is required but not installed."
        )

    self._model_name = model_name
    self._nbr_lora_adapters = 0
    
    # Initialize Unsloth model
    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
        **kwargs
    )
    # Ensure the model is a PeftModel to support multiple adapters correctly
    if not hasattr(self.model, "peft_config"):
        print("Converting base model to PeftModel for multi-adapter support.")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
        )
    
    # Ensure pad_token is set (crucial for Llama-3 models in Unsloth)
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

    if self.tokenizer.chat_template is None or "Qwen" in self._model_name or "Minitaur" in self._model_name:
        print("Using custom ChatML template.")
        self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

  def finalize_inference(self):
    """Call after loading all LoRA adapters to optimize for inference."""
    self.model = FastLanguageModel.for_inference(self.model)

  def increment_lora_adapters(self) -> int:
    self._nbr_lora_adapters += 1
    return self._nbr_lora_adapters
    
  def load_adapter(self, lora_path: str, adapter_name: str):
      self.model.load_adapter(lora_path, adapter_name)
    
  def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
      return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = DEFAULT_TERMINATORS,
      temperature: float = DEFAULT_TEMPERATURE,
      top_p: float = DEFAULT_TOP_P,
      top_k: int = DEFAULT_TOP_K,
      timeout: float = DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      adapter_name: str | None = None,
  ) -> str:
    del timeout
    
    if adapter_name:
        self.model.set_adapter(adapter_name)
    
    inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Ensure inputs are in the correct dtype if the model has embedding layers
    # that might not handle default LongTensor if they were converted (unlikely for IDs, but safe for others)
    # However, the error is in the linear layer (mat1 and mat2).
    # mat1 is hidden_states, mat2 is lm_head.weight.
    
    prompt_len = inputs.input_ids.shape[1]
    
    do_sample = temperature > 0
    
    # Get all potential stop token IDs
    stop_tokens = [self.tokenizer.eos_token_id]
    # Only add <|im_end|> as a stop token ID if it exists in the original vocabulary
    # to avoid "out of bounds" errors if it's not a single token.
    # For some models like Gemma-3, self.tokenizer might be a Processor.
    if hasattr(self.tokenizer, "vocab"):
        im_end_id = self.tokenizer.vocab.get("<|im_end|>", None)
    elif hasattr(self.tokenizer, "get_vocab"):
        im_end_id = self.tokenizer.get_vocab().get("<|im_end|>", None)
    else:
        im_end_id = None
        
    if im_end_id is not None:
        stop_tokens.append(im_end_id)

    # Use StoppingCriteria to handle cases where the model generates the stop string as multiple tokens
    stopping_criteria = StoppingCriteriaList()
    if "<|im_end|>" not in (terminators or []):
        stopping_criteria.append(StopOnString(self.tokenizer, "<|im_end|>", prompt_len))
    
    if terminators:
        for term in terminators:
            stopping_criteria.append(StopOnString(self.tokenizer, term, prompt_len))
    
    # Use autocast to ensure operations are performed in bfloat16
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=stop_tokens,
                stopping_criteria=stopping_criteria,
            )
        
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    if terminators:
        for term in terminators:
            if term in generated_text:
                generated_text = generated_text.split(term)[0]

    return generated_text

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
      adapter_name: str | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
      
    if adapter_name:
        self.model.set_adapter(adapter_name)
        
    # Pre-tokenize prompt once to get its length
    prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    logprobs = []
    with torch.no_grad():
        for response in responses:
            full_text = prompt + response
            inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")
            
            # Use autocast to ensure operations are performed in bfloat16
            # This is specifically needed for models like Gemma-3 where the unsloth compiler 
            # might have internal float32/bfloat16 mismatches in the forward pass.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**inputs)
            logits = outputs.logits # (1, seq_len, vocab_size)
            
            # Calculate log-probs for the ENTIRE sequence once to ensure alignment
            # Logits at index i predict tokens at index i+1
            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = inputs.input_ids[0, 1:].contiguous()
            
            # Safety check: ensure lengths match before gather
            # This handles cases where unsloth might return slightly different shapes
            min_len = min(shift_logits.shape[0], shift_labels.shape[0])
            shift_logits = shift_logits[-min_len:]
            shift_labels = shift_labels[-min_len:]
            
            log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs_all = log_probs_all.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            
            # Identify response tokens (those generated after the prompt)
            num_resp_tokens = inputs.input_ids.shape[1] - prompt_len
            if num_resp_tokens <= 0:
                # If prompt/response merging happened, use at least the last token
                # or 0 if it was truly empty
                num_resp_tokens = max(0, min(1, token_log_probs_all.shape[0]))
            
            # Take log-probs from the end of the combined sequence
            response_log_probs = token_log_probs_all[-num_resp_tokens:] if num_resp_tokens > 0 else torch.tensor([], device="cuda", dtype=token_log_probs_all.dtype)
            
            total_logprob = response_log_probs.sum().item()
            logprobs.append(total_logprob)
            
    best_idx = int(max(range(len(logprobs)), key=lambda i: logprobs[i]))
    
    debug_info = {
        'logprobs': {response: logprobs[i]
                     for i, response in enumerate(responses)},
        'method': 'logprobs'
    }
    
    return best_idx, responses[best_idx], debug_info


class UnslothLora:
  """Language model wrapper for Unsloth local inference with LoRA."""

  def __init__(
      self,
      model_name: str,
      *,
      lora_path: str,
      unsloth_language_model: UnslothLanguageModel | None = None,
      **kwargs: Any,
  ):
    self.lora_path = lora_path
    if unsloth_language_model is not None:
        self._base_model = unsloth_language_model
    else:
        self._base_model = UnslothLanguageModel(model_name, **kwargs)
        
    self._adapter_name = f"adapter_{self._base_model.increment_lora_adapters()}"
    print(f"Loading LoRA adapter from {lora_path} into adapter slot {self._adapter_name}")
    self._base_model.load_adapter(lora_path, adapter_name=self._adapter_name)

  def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
      return self._base_model.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = DEFAULT_TERMINATORS,
      temperature: float = DEFAULT_TEMPERATURE,
      top_p: float = DEFAULT_TOP_P,
      top_k: int = DEFAULT_TOP_K,
      timeout: float = DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    return self._base_model.sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
        adapter_name=self._adapter_name,
    )

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    return self._base_model.sample_choice(
        prompt,
        responses,
        seed=seed,
        adapter_name=self._adapter_name,
    )
