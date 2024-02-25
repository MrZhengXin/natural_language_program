import json
import argparse
from os import error
from time import sleep
import random
from tqdm import tqdm

from language_model_api import GPT3, ChatGPT

random.seed()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4')
parser.add_argument('--src', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--stop', type=str, default=None, nargs='+')
parser.add_argument('--max_try', type=int, default=8)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--disable_eos', action='store_true')
parser.add_argument('--max_tokens', type=int, default=3333)
parser.add_argument('--max_instance', type=int, default=8888888888)
parser.add_argument('--token', type=str, default='')
args = parser.parse_args()

if args.model in ['text-davinci-003']:
    llm = GPT3() if args.token=='' else GPT3(token=args.token)
elif args.model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-32k']:
    llm = ChatGPT(model=args.model, token=args.token)
else:
    raise AssertionError



with open(args.src, 'r') as f:
    test_instance_list = f.readlines()
test_instance_list = [json.loads(test_instance) for test_instance in test_instance_list]

cnt_previous = 0
try:
    with open(args.tgt, 'r') as f:
        cnt_previous = len(f.readlines())
except:
    pass

total_pred = 0
cnt_correct = 0
last_n_lines = 3

if not args.model.startswith('gpt-4'):
    args.max_tokens = 2700

fw = open(args.tgt, 'a')
for test_instance in tqdm(test_instance_list[cnt_previous:args.max_instance]):
    total_pred += 1
    output = ''
    prompt = test_instance['input']

    max_tokens = args.max_tokens
    previous_output = ''
    
    for _ in range(args.max_try):
        try:

            output, error = llm.infer_stream(prompt, temperature=args.temperature, stop=args.stop, max_tokens=max_tokens, top_p=args.top_p, disable_eos=args.disable_eos)

            if "\n\nThe final" in output:
                break
            if "and so on..." in output or "Continue the procedure" in output or "Continue the steps until" in output:
                # output = re.sub("\(?Repeat.*", '', output).strip()
                print("\nThe model skip steps.")
                continue
                # error = True
            if error: 
                print("Incomplete output")
                # if not error and re.match(r'.*\n+59\.[^\n]*\n+$', output, re.DOTALL):
                    # output += "60."
                prompt += output
                max_tokens = 3000
                previous_output += output
                continue
            
            else:
                break
            
        except Exception as e: 
            print(e)
            if "This model's maximum context length is" in str(e):
                max_tokens = max_tokens // 2
            print("API failure")
            sleep(10)
            continue
    output = previous_output + output    
    print(json.dumps(output), file=fw)
    print(prompt)
    print(output)
    if 'output' in test_instance.keys():
        print(test_instance['output'])
        if '\n' not in test_instance['output']:
            correct = str(test_instance['output']) in output.split('\n')[-1]
        else:
            correct = test_instance['output'] in output
        print(correct)
