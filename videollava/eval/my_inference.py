import math
import os
os.environ['RANK'] = str(0)
os.environ['CUDA_VISIBLE_DEVICES']='0'

import argparse
import json
import random

import sys
sys.path.append('/data/codes/lixiang/Video-LLaVA-main')

import torch
import transformers
import numpy as np
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


from rouge_score import rouge_scorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

"""
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 deepspeed /data/codes/lixiang/Video-LLaVA-main/videollava/eval/my_inference.py --model_path /data/codes/lixiang/Video-LLaVA-main/checkpoints/videollava-7b_my_finetune_matchtime_0909_2_epoch10/checkpoint-2600 --cache_dir /data/codes/lixiang/Video-LLaVA-main/cache_dir/ --video_dir /data/codes/lixiang/soccernet-matchtime/ --gt_file /data/codes/lixiang/Video-LLaVA-main/dataset/soccernet_json/soccernet_finetune_matchtime_eval.json --output_dir /data/codes/lixiang/Video-LLaVA-main/dataset/ 
"""



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def compute_bleu(reference, hypothesis):
    # reference and hypothesis should be tokenized lists of words
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothing_function)

def compute_rouge(reference, hypothesis):
    # Initialize a rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(' '.join(reference), ' '.join(hypothesis))
    return scores

def compute_meteor(reference, hypothesis):
    # Meteor metric from pycocoevalcap
    meteor_scorer = Meteor()
    score, _ = meteor_scorer.compute_score({0: [reference]}, {0: [hypothesis]})
    return score

def compute_cider(reference, hypothesis):
    # CIDEr metric from pycocoevalcap
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score({0: [reference]}, {0: [hypothesis]})
    return score



def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file', required=True)
    # parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', default='video_qa_pred_res', help='Name of the file for storing results JSON.')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # print(video_tensor.shape)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor], do_sample=True, temperature=0.7, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    # Load both ground truth file containing questions and answers
    # with open(args.gt_file_question) as file:
    #     gt_questions = json.load(file)
    # with open(args.gt_file_answers) as file:
    #     gt_answers = json.load(file)

    # gt_questions = json.load(open(args.gt_file_question, "r"))
    # gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    # gt_answers = json.load(open(args.gt_file_answers, "r"))
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)
    gt_samples = json.load(open(args.gt_file, 'r'))

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    mean_bleu, mean_rouge, mean_meteor, mean_cider = [], [], [], []
    for sample in tqdm(gt_samples):
        video_name = sample['video']
        question = sample['conversations'][0]['value']
        id = sample['id']
        answer = sample['conversations'][1]['value']
        index += 1

        sample_set = {'video': video_name ,'id': id, 'question': question, 'answer': answer}

        # Load the video file
        # for fmt in tqdm(video_formats):  # Added this line
        temp_path = os.path.join(args.video_dir, f"{video_name}")
        if os.path.exists(temp_path):
            video_path = temp_path
            # try:
            # Run inference on the video and add the output to the list
            try:
                output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
            except Exception as E:
                print(E)
                continue
            sample_set['pred'] = output
            output_list.append(sample_set)
            # except Exception as e:
            #     print(f"Error processing video file '{video_name}': {e}")

            # reference = answer.split() 
            # hypothesis = output.split()
            # bleu_score = compute_bleu(reference, hypothesis)
            # rouge_scores = compute_rouge(reference, hypothesis)
            # meteor_score = compute_meteor('.'.join(reference), '.'.join(hypothesis))
            # cider_score = compute_cider('.'.join(reference), '.'.join(hypothesis))

            # Add the metrics to the sample set
            # sample_set['bleu'] = bleu_score
            # sample_set['rouge'] = rouge_scores
            # sample_set['meteor'] = meteor_score
            # sample_set['cider'] = cider_score

            # mean_bleu.append(bleu_score)
            # mean_rouge.append(rouge_scores['rougeL'][2])  # rougeL, fmeasure
            # mean_meteor.append(meteor_score)
            # mean_cider.append(cider_score)
            # print(np.mean(mean_bleu), np.mean(mean_rouge), np.mean(mean_meteor), np.mean(mean_cider))


            ans_file.write(json.dumps(sample_set) + "\n")


        #     # break
        # if index >:
        #     break

    ans_file.close()
    
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
