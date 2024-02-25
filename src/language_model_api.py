from abc import ABCMeta, abstractmethod
import openai

class LanguageModelAPI:
    __metaclass__ = ABCMeta
    @abstractmethod
    def infer(self):
        pass

import tiktoken

def decide_logit_bias(first_line, model="gpt-4", disable_eos=False):
    logit_bias = {}
    # https://platform.openai.com/docs/api-reference/completions/create#completions/create-logit_bias
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    encoding = tiktoken.encoding_for_model(model)
    skip_words = ["...", "......", "....", ".....", "Repeat", "repeat", "…"]
    if model in ['gpt-4', 'gpt-4-32k']:
        skip_words.append("...\n\n")

    if "Do not jump steps." in first_line:
        for word in skip_words:
            logit_bias[encoding.encode(word)[0]] = -100
            logit_bias[encoding.encode(' ' + word)[0]] = -100

        # logit_bias = {986: -100, 16317: -100, 1106: -100, 12359: -100, 628: -100, 40322: -100, 1399: -100} 
        # 986: "...", 16317: "......", 1106: "....", 12359: ".....", 628: "\n\n", 40322: "Repeat", 1399: "…"
    if disable_eos:
        if logit_bias is None:
            logit_bias = {50256: -100} 
            # 50256: "<|endoftext|>"
        else:
            logit_bias[50256] = -100
    return logit_bias

class GPT3(LanguageModelAPI):
    def __init__(self, model="text-davinci-003", token="") -> None:
        openai.api_key = token
        self.model = model

    # @retry(wait=wait_random_exponential(min=60, max=120), stop=stop_after_attempt(6))
    def infer(self, prompt, temperature=0.7, max_tokens=3333, stop=None, top_p=1.0, disable_eos=False):
        logit_bias = decide_logit_bias(prompt.split('\n')[0], model=self.model, disable_eos=disable_eos)
        response = openai.Completion.create(engine=self.model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, stop=stop, request_timeout=180, top_p=top_p, logit_bias=logit_bias)
        output_text = response["choices"][0]["text"]
        return output_text

    def infer_stream(self, prompt, temperature=1, max_tokens=3100, stop=None, top_p=1.0, disable_eos=False):
        logit_bias = decide_logit_bias(prompt.split('\n')[0], model=self.model, disable_eos=disable_eos)
        response = openai.Completion.create(model=self.model, prompt=prompt, temperature=temperature, max_tokens=3000, stop=stop, request_timeout=180, top_p=top_p, logit_bias=logit_bias, stream=True)
        completion_text = ''
        # iterate through the stream of events
        error = False
        try:
            for event in response:
                try:
                    event_text = event['choices'][0]['text']  # extract the text
                except:
                    event_text = ''
                completion_text += event_text  # append the text
                print(event_text, end='')  # print the delay and text
        except Exception as e: 
            print(e)
            error = True
        return completion_text, error

    def infer_batch(self, prompt, temperature=0.7, max_tokens=3333, stop=None, top_p=1.0, disable_eos=False):
        logit_bias = decide_logit_bias(prompt[0].split('\n')[0], model=self.model, disable_eos=disable_eos)
        response = openai.Completion.create(engine=self.model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, stop=stop, request_timeout=600, top_p=top_p, logit_bias=logit_bias)
        output_text = [r["text"] for r in response["choices"]]
        return output_text

class ChatGPT(LanguageModelAPI):
    def __init__(self, model="gpt-3.5-turbo", token="") -> None:
        
        openai.api_key = token
        self.model = model

    # @retry(wait=wait_random_exponential(min=60, max=120), stop=stop_after_attempt(6))
    def infer(self, prompt, temperature=1, max_tokens=3100, stop=None, top_p=1.0, disable_eos=False):
        logit_bias = decide_logit_bias(prompt.split('\n')[0], model=self.model, disable_eos=disable_eos) if self.model != 'gpt-4' else {}
        response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], temperature=temperature, stop=stop, top_p=top_p, logit_bias=logit_bias, request_timeout=2000, max_tokens=max_tokens)
        output_text = response["choices"][0]["message"]["content"]
        return output_text

    def infer_stream(self, prompt, temperature=1, max_tokens=3100, stop=None, top_p=1.0, disable_eos=False):
        logit_bias = decide_logit_bias(prompt.split('\n')[0], model=self.model, disable_eos=disable_eos)
        response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=temperature, stop=stop, top_p=top_p, logit_bias=logit_bias, request_timeout=60, max_tokens=max_tokens, stream=True)
        completion_text = ''
        # iterate through the stream of events
        error = False
        try:
            for event in response:
                try:
                    event_text = event["choices"][0]["delta"]["content"]  # extract the text
                except:
                    event_text = ''
                completion_text += event_text  # append the text
                print(event_text, end='')  # print the delay and text
        except Exception as e: 
            print(e)
            error = True
        return completion_text, error

    # @retry(wait=wait_random_exponential(min=7, max=60), stop=stop_after_attempt(6))
    def infer_batch(self, prompt_list, temperature=1, max_tokens=1024, stop=None, top_p=1.0, disable_eos=False):
        logit_bias = decide_logit_bias(prompt_list[0].split('\n')[0], model=self.model, disable_eos=disable_eos)
        response = openai.ChatCompletion.create(model=self.model, messages=[[{"role": "user", "content": prompt}] for prompt in prompt_list], temperature=temperature, stop=stop, top_p=top_p)
        output_text = [r["choices"][0]["message"]["content"] for r in response]
        return output_text
