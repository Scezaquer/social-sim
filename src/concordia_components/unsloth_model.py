"""Language Model that uses Unsloth for local inference."""

from collections.abc import Collection, Sequence
from typing import Any, Mapping
import os
import re
import torch

from concordia.language_model import language_model
from concordia.utils.deprecated import measurements as measurements_lib
from typing_extensions import override

try:
  from unsloth import FastLanguageModel
  UNSLOTH_AVAILABLE = True
except ImportError:
  FastLanguageModel = None
  UNSLOTH_AVAILABLE = False

try:
    from peft import PeftConfig
    from peft.utils.save_and_load import set_peft_model_state_dict
    PEFT_AVAILABLE = True
except ImportError:
    PeftConfig = None
    set_peft_model_state_dict = None
    PEFT_AVAILABLE = False

try:
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    safe_load_file = None
    SAFETENSORS_AVAILABLE = False

from transformers import StoppingCriteria, StoppingCriteriaList

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

class UnslothLanguageModel(language_model.LanguageModel):
  """Language model wrapper for Unsloth local inference."""

  def __init__(
      self,
      model_name: str,
      *,
      max_seq_length: int = 4096,
      load_in_4bit: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      **kwargs: Any,
  ):
    if not UNSLOTH_AVAILABLE:
        raise ImportError(
            "Unsloth is required but not installed."
        )

    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._nbr_lora_adapters = 0
    
    # Initialize Unsloth model
    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        **kwargs
    )
    # Ensure pad_token is set (crucial for Llama-3 models in Unsloth)
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

    if self.tokenizer.chat_template is None or "Qwen" in self._model_name or "Minitaur" in self._model_name:
        print("Using custom ChatML template.")
        self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

  def finalize_inference(self):
    """Call after loading all LoRA adapters to optimize for inference."""
    FastLanguageModel.for_inference(self.model)

  def increment_lora_adapters(self) -> int:
    self._nbr_lora_adapters += 1
    return self._nbr_lora_adapters

    def _find_adapter_weights_file(self, lora_path: str) -> str | None:
        candidates = (
                "adapter_model.safetensors",
                "adapter_model.bin",
                "pytorch_model.bin",
        )
        for filename in candidates:
            full_path = os.path.join(lora_path, filename)
            if os.path.isfile(full_path):
                return full_path
        return None

    def _load_adapter_state_dict(self, weights_path: str) -> Mapping[str, torch.Tensor]:
        if weights_path.endswith(".safetensors"):
            if not SAFETENSORS_AVAILABLE:
                raise ImportError(
                        "safetensors is required to load adapter_model.safetensors"
                )
            return safe_load_file(weights_path)
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            return state_dict["state_dict"]
        return state_dict

    def _extract_adapter_tokens(self, state_dict: Mapping[str, torch.Tensor]) -> set[str]:
        tokens: set[str] = set()
        # Example key: model.layers.0.self_attn.q_proj.lora_A.adapter_1.weight
        pattern = re.compile(r"\.lora_[AB]\.([^.]+)\.weight$")
        for key in state_dict.keys():
            match = pattern.search(key)
            if match:
                tokens.add(match.group(1))
        return tokens

    def _remap_adapter_token(
            self,
            state_dict: Mapping[str, torch.Tensor],
            source_token: str,
            target_token: str,
    ) -> dict[str, torch.Tensor]:
        remapped: dict[str, torch.Tensor] = {}
        marker = f".{source_token}."
        replacement = f".{target_token}."
        for key, value in state_dict.items():
            remapped[key.replace(marker, replacement)] = value
        return remapped

    def _load_adapter_with_state_dict_fallback(self, lora_path: str, adapter_name: str) -> None:
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                    "PEFT is required for fallback adapter loading, but it is not available."
            )

        weights_path = self._find_adapter_weights_file(lora_path)
        if weights_path is None:
            raise FileNotFoundError(
                    f"No adapter weights file found under {lora_path}."
            )

        state_dict = self._load_adapter_state_dict(weights_path)
        adapter_tokens = self._extract_adapter_tokens(state_dict)

        if len(adapter_tokens) == 1:
            source_token = next(iter(adapter_tokens))
            if source_token != adapter_name:
                print(
                        f"Remapping checkpoint adapter token '{source_token}' to '{adapter_name}' "
                        f"for {lora_path}"
                )
                state_dict = self._remap_adapter_token(state_dict, source_token, adapter_name)

        peft_config = PeftConfig.from_pretrained(lora_path)
        if adapter_name not in getattr(self.model, "peft_config", {}):
            self.model.add_adapter(adapter_name, peft_config)

        set_peft_model_state_dict(self.model, state_dict, adapter_name=adapter_name)
        self.model.set_adapter(adapter_name)
    
  def load_adapter(self, lora_path: str, adapter_name: str):
      # Newer Unsloth/PEFT combinations can mismatch checkpoint adapter tokens
      # (e.g., checkpoint has adapter_1 while runtime expects adapter_2).
      # When possible, load via state_dict with token remapping.
      try:
          self._load_adapter_with_state_dict_fallback(lora_path, adapter_name)
      except Exception as remap_exc:
          print(
              f"Fallback adapter load failed for {lora_path} ({remap_exc}). "
              "Trying native load_adapter."
          )
          self.model.load_adapter(lora_path, adapter_name)
    
  def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
      return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
      
  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      adapter_name: str | None = None,
  ) -> str:
    del timeout
    
    if adapter_name:
        self.model.set_adapter(adapter_name)
    
    inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[1]
    
    do_sample = temperature > 0
    
    # Get all potential stop token IDs
    stop_tokens = [self.tokenizer.eos_token_id]
    # Only add <|im_end|> as a stop token ID if it exists in the original vocabulary
    # to avoid "out of bounds" errors if it's not a single token.
    im_end_id = self.tokenizer.vocab.get("<|im_end|>", None)
    if im_end_id is not None:
        stop_tokens.append(im_end_id)

    # Use StoppingCriteria to handle cases where the model generates the stop string as multiple tokens
    stopping_criteria = StoppingCriteriaList()
    if "<|im_end|>" not in (terminators or []):
        stopping_criteria.append(StopOnString(self.tokenizer, "<|im_end|>", prompt_len))
    
    if terminators:
        for term in terminators:
            stopping_criteria.append(StopOnString(self.tokenizer, term, prompt_len))
    
    with torch.no_grad():
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
                
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(generated_text)},
      )

    return generated_text

  @override
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
            response_log_probs = token_log_probs_all[-num_resp_tokens:] if num_resp_tokens > 0 else torch.tensor([], device="cuda")
            
            total_logprob = response_log_probs.sum().item()
            logprobs.append(total_logprob)
            
    best_idx = int(max(range(len(logprobs)), key=lambda i: logprobs[i]))

    if self._measurements is not None:
        self._measurements.publish_datum(
            self._channel,
            {'choice_method': 'logprobs', 'num_choices': len(responses)},
        )
    
    debug_info = {
        'logprobs': {response: logprobs[i]
                     for i, response in enumerate(responses)},
        'method': 'logprobs'
    }
    
    return best_idx, responses[best_idx], debug_info


class UnslothLora(language_model.LanguageModel):
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

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
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

  @override
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
