import shutil
import subprocess
import sys
sys.path.append('/data/codes/lixiang/Video-LLaVA-main')

import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
from decord import VideoReader, cpu
from transformers import TextStreamer

from videollava.constants import DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle, Conversation
from videollava.serve.gradio_utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css
from videollava.eval.my_inference import *

from videollava.constants import IMAGE_TOKEN_INDEX
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_token
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init


def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    # print(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename


class SoccerCommentChat:
    def __init__(self, model_path, conv_mode, model_base=None, load_8bit=False, load_4bit=False, device='cuda', cache_dir=None):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                                   load_8bit, load_4bit,
                                                                                   device=device, cache_dir=cache_dir)
        self.image_processor = processor['image']
        self.video_processor = processor['video']
        self.conv_mode = conv_mode
        self.conv = conv_templates[conv_mode].copy()
        self.device = self.model.device
        print(self.model)

    def get_prompt(self, qs, state):
        state.append_message(state.roles[0], qs)
        state.append_message(state.roles[1], None)
        return state

    @torch.inference_mode()
    # def generate(self, images_tensor: list, prompt: str, first_run: bool, state):
    def generate(self, video_tensor, qs, first_run, state):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN] * 8) + DEFAULT_VID_END_TOKEN + '\n' + qs
        elif first_run:
            qs = ''.join([DEFAULT_IMAGE_TOKEN] * 8) + '\n' + qs

        conv_mode = "llava_v1"
        args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        state = self.get_prompt(qs, state)
        prompt = state.get_prompt()




        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            args.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=[video_tensor], do_sample=True, temperature=0.2, max_new_tokens=1024,
                                        use_cache=True, stopping_criteria=[stopping_criteria], streamer=streamer)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs, state


def generate(video, textbox_in, first_run, state, state_, video_tensor):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    video = video if video else "none"
    # assert not (os.path.exists(image1) and os.path.exists(video))

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        # images_tensor = []

    first_run = False if len(state.messages) > 0 else True
    text_en_in = textbox_in
    if first_run:
        video_tensor = handler.video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    text_en_out, state_ = handler.generate(video_tensor, text_en_in, first_run, state)   # get_model_output(handler.model, video_processor, handler.tokenizer, video, 'What is happening in this video?', args)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    show_images = ""
    # if os.path.exists(image1):
    #     filename = save_image_to_local(image1)
    #     show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if os.path.exists(video):
        filename = save_video_to_local(video)
        show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

    state.messages.pop(-1)
    state.messages.pop(-1)
    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    state.append_message(state.roles[1], textbox_out)

    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), gr.update(value=video if os.path.exists(video) else None, interactive=True), video_tensor)


def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True), \
            True, state, state_, state.to_gradio_chatbot())


conv_mode = "v1_soccerComment"
# model_path = '/data/lx/Video-LLaVA-main/checkpoints/videollava-7b_my_finetune_0802' #'LanguageBind/Video-LLaVA-7B'
cache_dir = './cache_dir'
device = 'cuda:0'
# load_8bit = True
# load_4bit = False
dtype = torch.float16
# handler = Chat(model_path, conv_mode=conv_mode, load_8bit=load_8bit, load_4bit=load_8bit, device=device, cache_dir=cache_dir)
# handler.model.to(dtype=dtype)
# if not os.path.exists("temp"):
#     os.makedirs("temp")

# --------------inference-------------
args = parse_args()
handler = SoccerCommentChat(args.model_path, conv_mode=conv_mode, load_8bit=False, load_4bit=False, device=device, cache_dir=cache_dir)
# model_name = get_model_name_from_path(args.model_path)
# tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
# model = model.to(args.device)
soccer_title_markdown = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <div>
    <h1 > ⚽️ SoccerComment: Multi-Modal Large Language Model with RAG Strategies in Soccer Commentary Generation</h1>
  </div>
</div>
""")

app = FastAPI()


textbox = gr.Textbox(
    show_label=False, placeholder="Enter text and press ENTER", container=False
)
videopath = gr.Textbox(
    show_label=False, placeholder="Video Path", container=False
)
with gr.Blocks(title='SoccerComment⚽️', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(soccer_title_markdown)
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()
    video_tensor = gr.State()


    with gr.Row():
        with gr.Column(scale=3):
            # image1 = gr.Image(label="Input Image", type="filepath")
            video = gr.Video(label="Input Video")


            cur_dir = os.path.dirname(os.path.abspath(__file__))

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Video-LLaVA", bubble_full_width=True).style(height=750)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                # with gr.Column(scale=8):
                #     videopath.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=True
                    )
                # output = gr.Textbox(label="commentary")
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)


    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)
    # video_path = '/data/lx/soccernet/caption_anno_clips/germany_bundesliga_2016-2017_2016-10-01_-_19-30_Bayer_Leverkusen_2_-_0_Dortmund_2_720p_clip_19.mp4'
    # question = '<video>\nWhat is happening in this video?'
    # submit_btn.click(generate1, [videopath, textbox], [output])
    submit_btn.click(generate, [video, textbox, first_run, state, state_, video_tensor],
                     [state, state_, chatbot, first_run, textbox, video, video_tensor])

    regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
        generate, [video, textbox, first_run, state, state_], [state, state_, chatbot, first_run, textbox, video])
    #
    clear_btn.click(clear_history, [state, state_],
                    [video, textbox, first_run, state, state_, chatbot])

# app = gr.mount_gradio_app(app, demo, path="/")
demo.launch(share=True)

# uvicorn videollava.serve.gradio_web_server:app
# python -m  videollava.serve.gradio_web_server
