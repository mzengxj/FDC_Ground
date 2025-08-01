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
from process_utils import pred_2_point, extract_bbox, completion2point
from qwen_vl_utils import process_vision_info
import PIL
from PIL import Image
import pandas as pd


logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--lora_path', type=str, required=True)
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
args = parser.parse_args()

qwen_path = args.qwen_path
#tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)

if args.lora_path != 'Qwen-VL-Chat':
    # use lora
    lora_path = args.lora_path
    model = AutoPeftModelForCausalLM.from_pretrained(lora_path, device_map="cuda", trust_remote_code=True).eval()
else:
    # use Qwen-VL-Chat
    model_path = qwen_path
    #model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()

print("Load Success")
pad_token_id = processor.tokenizer.pad_token_id
#generation_config = GenerationConfig(
#            max_new_tokens=10000,
#            do_sample=True,
#            temperature=1,  # HACK
#            num_return_sequences=1,
#            pad_token_id=pad_token_id,
#        )
##model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)
#model.generation_config = generation_config

if args.task == "all":
    tasks = ["mobile", "desktop", "web"]
else:
    tasks = [args.task]
tasks_result = []
#result = []
for task in tasks:
    dataset = "screenspot_" + task + ".json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print("Num of sample: " + str(len(screenspot_data)))
    #prompt_origin = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"
    #prompt_origin = "In this UI screenshot, what is the precise coordinates (x, y) of the element corresponding to the command \"{}\""
    #rompt_origin = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"
    prompt_origin = "Output the bounding box in the image corresponding to the instruction \"{}\" with grounding."
    prompt_origin_qwen = "Generate the bounding box of {}"
    prompt_grpo =  prompt_origin + "  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0
    
    result_data = []

    for j, item in tqdm(enumerate(screenspot_data)):
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        image = Image.open(img_path)
        instruction = item["instruction"]
        #query = prompt_origin.format(instruction)
        query = prompt_grpo.format(instruction)
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img_size = image.size
        #bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
        bbox = [bbox[0] , bbox[1] , bbox[2] , bbox[3] ]
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
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        #image_inputs, video_inputs = process_vision_info(messages)
        img = PIL.Image.open(img_path)
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
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        #prompt = prompt_origin.format(instruction)
        
        #query = tokenizer.from_list_format([{'image': img_path},  # Either a local path or an url
        #                                    {'text': prompt}, ])
        # print(query)
        #response, history = model.chat(tokenizer, query=query, history=None)
        # print(response)

        #try:
        #if 'box' in response:
        #    pred_bbox = extract_bbox(response)
        #    click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
        #    click_point = [item / 1000 for item in click_point]
        #else:
        #click_point = pred_2_point(output_text[0]) not recognize all number
        click_point = completion2point(output_text[0])
        item['answer'] = output_text[0]
        if click_point is not None  and (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
            corr_action += 1
            item['corr_action'] = 1
            if item["data_type"] == 'text':
                text_correct.append(1)
            else:
                icon_correct.append(1)
            logging.info("match " + str(corr_action / num_action))
        else:
            item['corr_action'] = 0
            if item["data_type"] == 'text':
                text_correct.append(0)
            else:
                icon_correct.append(0)
            logging.info("unmatch " + str(corr_action / num_action))

        result_data.append(item)
        #result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "answer": output_text[0], "pred": click_point,
        #               "type": item["data_type"], "source": item["data_source"]})
        #except:
        #    num_wrong_format += 1
        #    if item["data_type"] == 'text':
        #        text_correct.append(0)
        #    else:
        #        icon_correct.append(0)
        #    logging.info("Step: " + str(j) + " wrong format")
    write_file = "result_" + task + ".json"
    pd.DataFrame(result_data).to_json(os.path.join(args.screenspot_test, write_file))




    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
    logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

    text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
    icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
    tasks_result.append([text_acc, icon_acc])

logging.info(tasks_result)

