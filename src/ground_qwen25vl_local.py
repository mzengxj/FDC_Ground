import argparse
import os
import json
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from PIL import Image
import base64
from openai import AsyncOpenAI
from asyncio import Semaphore
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import ast
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import PIL
from PIL import Image
import pandas as pd


logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

def completion2point(s):
    click_point = None
    regex = r"<think>(.*?)</think>\s*<answer>(.*)</answer>"
    match = re.search(regex, s, re.DOTALL)
    if not match:
        return click_point

    think, answer = match.group(1), match.group(2)
    numbers = re.findall(r'\d+', answer)
    numbers = [float(num) for num in numbers]
    if len(numbers) >= 2:
        click_point = numbers[:2]
    return click_point


def eval_model(args):
    """Evaluate the model."""
    qwen_path = args.model_path
    #tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, device_map="cuda", trust_remote_code=True).eval()
    print("Load Success")
    pad_token_id = processor.tokenizer.pad_token_id
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    prompt_origin = "Output the bounding box in the image corresponding to the instruction \"{}\" with grounding."
    prompt_grpo =  prompt_origin + "  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    results = []
    with open(answers_file, "w") as ans_file:
        for line in questions:
            if args.image_folder:
                image_base_dir = os.path.expanduser(args.image_folder)
                image_path = os.path.join(image_base_dir, line[args.image_key])
            else:
                image_path = os.path.expanduser(line[args.image_key])

            instruction = line["description"]
            query = prompt_grpo.format(instruction)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            #"image": img_path,
                        },
                        {"type": "text", "text": query},
                    ],
                }
             ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            #image_inputs, video_inputs = process_vision_info(messages)
            img = PIL.Image.open(image_path)
            #breakpoint()
            inputs = processor(
                text=[text],
                images=[img],
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            #inputs = processor(
            #    text=[text],
            #    images=image_inputs,
            #    videos=video_inputs,
            #    padding=True,
            #    return_tensors="pt",
            #)
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=1000, temperature=args.temperature)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)
            logging.info(output_text[0])
            click_point = completion2point(output_text[0])
#             response_text = output_text
#             try:
#                 ratio_coords = eval(response_text)
#             except:
#                 numbers = re.findall(r'\d+', response_text)
#                 ratio_coords = (int(numbers[0]), int(numbers[1]))

            try:
                x_ratio, y_ratio = click_point
            except:
                x_ratio, y_ratio = -1, -1

            # Convert to absolute coordinates
            #x_coord = int(x_ratio / 1000 * width)
            #y_coord = int(y_ratio / 1000 * height)
            x_coord = int(x_ratio)
            y_coord = int(y_ratio)

            line["output"] = f"({x_coord}, {y_coord})"
            line["model_id"] = os.path.expanduser(args.model_path)
            line["scale"] = 1.0

            results.append(line)
            ans_file.write(json.dumps(line) + "\n")
            print(json.dumps(line))

    #with open(answers_file, "w") as ans_file:
    #    for line in results:
    #        ans_file.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="osunlp/UGround-V1-7B")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="img_filename")
    parser.add_argument("--temperature", type=float, default=0)
    
    args = parser.parse_args()

    eval_model(args)
