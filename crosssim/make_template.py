"""Write the chat template the vLLM server uses for chat requests (OASIS, SiliSocS).

Same rule as the paper's simulator (unsloth_model.py): ChatML for Qwen/Minitaur
or any tokenizer without a template; otherwise the tokenizer's own template.
For ChatML we use a tool-aware variant (identical to plain ChatML when no tools
are passed; with tools it renders them Hermes-style, which is Qwen2.5's native
tool format), robust to None content and role="tool" messages.

Prints the tool-call parser name to use (hermes for ChatML, llama3_json for the
native Llama-3 template, none otherwise).
"""
import sys

CHATML_TOOLS = r"""{%- set sys_msg = messages[0]['content'] if messages and messages[0]['role'] == 'system' else none -%}
{%- if tools -%}
<|im_start|>system
{% if sys_msg %}{{ sys_msg }}

{% endif %}# Tools

You may call one or more functions. You are provided with function signatures within <tools></tools> XML tags:
<tools>{% for tool in tools %}
{{ tool | tojson }}{% endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{% elif sys_msg -%}
<|im_start|>system
{{ sys_msg }}<|im_end|>
{% endif -%}
{%- for message in messages -%}
{%- if loop.first and message['role'] == 'system' -%}
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{% if message['content'] %}{{ message['content'] }}{% endif %}
{%- if message.get('tool_calls') -%}{%- for tc in message['tool_calls'] -%}
{%- set fn = tc['function'] if tc.get('function') else tc %}
<tool_call>
{"name": "{{ fn['name'] }}", "arguments": {% if fn['arguments'] is string %}{{ fn['arguments'] }}{% else %}{{ fn['arguments'] | tojson }}{% endif %}}
</tool_call>
{%- endfor -%}{%- endif %}<|im_end|>
{% elif message['role'] == 'tool' -%}
<|im_start|>user
<tool_response>
{{ message['content'] if message['content'] is string else (message['content'] | tojson) }}
</tool_response><|im_end|>
{% else -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] if message['content'] is string else (message['content'] | tojson) }}<|im_end|>
{% endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif -%}"""


def main():
    hf_model, out_path = sys.argv[1], sys.argv[2]
    if hf_model == "dummy":  # tests only
        open(out_path, "w").write(CHATML_TOOLS)
        print("hermes")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_model, local_files_only=True)
    if tok.chat_template is None or "Qwen" in hf_model or "Minitaur" in hf_model:
        template, parser = CHATML_TOOLS, "hermes"
    else:
        template = tok.chat_template
        parser = "llama3_json" if "Llama-3" in hf_model or "llama" in hf_model.lower() else "none"
    with open(out_path, "w") as f:
        f.write(template)
    print(parser)


if __name__ == "__main__":
    main()
