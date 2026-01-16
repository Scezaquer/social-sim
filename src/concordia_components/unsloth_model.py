"""Language Model that uses Unsloth for local inference."""

from collections.abc import Collection, Sequence
from typing import Any, Mapping
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

class UnslothLanguageModel(language_model.LanguageModel):
  """Language model wrapper for Unsloth local inference."""

  def __init__(
      self,
      model_name: str,
      *,
      max_seq_length: int = 4096,
      load_in_4bit: bool = True,
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
    FastLanguageModel.for_inference(self.model)

  def increment_lora_adapters(self) -> int:
    self._nbr_lora_adapters += 1
    return self._nbr_lora_adapters
    
  def load_adapter(self, lora_path: str, adapter_name: str):
      self.model.load_adapter(lora_path, adapter_name)
      
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
    
    do_sample = temperature > 0
    
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
            eos_token_id=self.tokenizer.eos_token_id,
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
        
    logprobs = []
    with torch.no_grad():
        for response in responses:
            full_text = prompt + response
            inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")
            
            # Find where response starts
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_inputs.input_ids.shape[1]
            
            outputs = self.model(**inputs)
            logits = outputs.logits # (1, seq_len, vocab_size)
            
            # Shift logits
            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = inputs.input_ids[0, 1:].contiguous()
            
            start_idx = prompt_len - 1
            if start_idx < 0: start_idx = 0
            
            response_logits = shift_logits[start_idx:]
            response_labels = shift_labels[start_idx:]
            
            log_probs_tensor = torch.nn.functional.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs_tensor.gather(1, response_labels.unsqueeze(1)).squeeze(1)
            
            total_logprob = token_log_probs.sum().item()
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
    if unsloth_language_model is not None:
        self._base_model = unsloth_language_model
    else:
        self._base_model = UnslothLanguageModel(model_name, **kwargs)
        
    self._adapter_name = f"adapter_{self._base_model.increment_lora_adapters()}"
    self._base_model.load_adapter(lora_path, self._adapter_name)

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
