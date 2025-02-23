from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from faster_whisper import WhisperModel
import torch
from xcodec2.modeling_xcodec2 import XCodec2Model
import torchaudio
import gradio as gr
import numpy
import random

"""
#Llasa-8B 下载需要hf的token，下面是获取token和登录，如果需要就去除掉注释
#建议直接将模型下载到本地加载
import os
api_key = os.getenv("HF_TOKEN")

from huggingface_hub import login
login(token=api_key) 

"""


def set_seed(seed=49):
    numpy.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#fastwhisper
fastwhisper_path = "Systran/faster-whisper-large-v3"
fastwhisper_model = WhisperModel(fastwhisper_path, device="cpu")
language = None
def fastwhisper_asr_file(audio_file):
    text=""
    try:
        segments, info = fastwhisper_model.transcribe(
        audio          = audio_file,
        beam_size      = 5,
        vad_filter     = True,
        vad_parameters = dict(min_silence_duration_ms=700),
        language       = language)
        for segment in segments:
            text+=segment.text+"\n"
    except Exception as e:
        print("error"+str(e))
    #print(text)
    return text

#llasa8b
model_path = "HKUSTAudio/xcodec2"
Codec_model = XCodec2Model.from_pretrained(model_path,device_map='cuda')
Codec_model.eval()
llasa_8b ='HKUSTAudio/Llasa-8B'
tokenizer = AutoTokenizer.from_pretrained(llasa_8b)
llasa_model = AutoModelForCausalLM.from_pretrained(
    llasa_8b,
    torch_dtype=torch.float16,
    device_map='cuda'
)
llasa_model.eval()  


def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def tts(sample_audio_path,sample_text, system_prompt_text,target_text,sample_rate_input=44100,penalty=1.2,temperature=1.0,top_k=50,top_p=0.9,do_sample=True, random_seed=49):
    set_seed(random_seed)
    progress=gr.Progress()
    progress(0, '加载音频...')
    waveform, sample_rate = torchaudio.load(sample_audio_path)
    if len(waveform[0])/sample_rate > 15:
        gr.Warning("Trimming audio to first 15secs.")
        waveform = waveform[:, :sample_rate*15]
    # Check if the audio is stereo (i.e., has more than one channel)
    if waveform.size(0) > 1:
        # Convert stereo to mono by averaging the channels
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
    else:
        # If already mono, just use the original waveform
        waveform_mono = waveform
    prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate_input)(waveform_mono)
    progress(0.2, '处理参考文字...')
    if len(target_text) == 0:
        return None
    elif len(target_text) > 300:
        gr.Warning("要生成的文字文字太长了，自动截断到300个字符。")
        target_text = target_text[:300]
    input_text = sample_text + ' ' + target_text
    with torch.no_grad():
        progress(0.4, 'Codec编码...')
        vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
        vq_code_prompt = vq_code_prompt[0,0,:]
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": system_prompt_text +":" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]
        input_ids = tokenizer.apply_chat_template(
            chat, 
            tokenize=True, 
            return_tensors='pt', 
            continue_final_message=True
        )
        input_ids = input_ids.cuda()
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        progress(0.6, '生成音频...')
        outputs = llasa_model.generate(
                input_ids,
                max_length=2048,  # We trained our model with a max length of 2048
                eos_token_id= speech_end_id ,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,           
                temperature=temperature,
                repetition_penalty=penalty
                
            )
        generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True) 
        speech_tokens = extract_speech_ids(speech_tokens)
        speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
        progress(0.8, '解码音频...')
        gen_wav = Codec_model.decode_code(speech_tokens) 
        gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]
        progress(1, '完成!')
        return (sample_rate_input, gen_wav[0, 0, :].cpu().numpy())

#全局设置,在页面最上方
with gr.Blocks() as app_setting:
    gr.Markdown("## Settings")
    with gr.Row():
        with gr.Column():
            random_seed = gr.Number(label="Random Seed", value=49, minimum=0, maximum=10000000, step=1)
            sample_rate = gr.Dropdown(choices=[16000, 22050, 24000,28000,32000], value=24000, label="Sample Rate")
            do_sample = gr.Checkbox(label="Do Sample", value=True)
        with gr.Column():
            temperature = gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=1.0, step=0.05)
            top_k = gr.Slider(label="Top K", value=50, minimum=1, maximum=100, step=1)
            top_p = gr.Slider(label="Top P", value=1, minimum=0.0, maximum=1.0, step=0.05)
            penalty= gr.Slider(label="Penalty", value=1.2, minimum=0.0, maximum=2.0, step=0.05)                

    



with gr.Blocks() as app_tts:
    gr.Markdown("## 参考音频 .wav ")
    gr.Markdown("* 手动输入参考音频中的文字，或者使用 FastWhisper 进行自动转录")
    with gr.Row():
        with gr.Column():
            ref_audio_input = gr.Audio(label="参考音频", type="filepath")
        with gr.Column():
            ref_text_input = gr.Textbox(label="参考文本", lines=5, interactive=True)
            transcribe_btn = gr.Button("转录", variant="primary")  
        transcribe_btn.click(
            fn=fastwhisper_asr_file,
            inputs=[ref_audio_input],
            outputs=[ref_text_input]
        )
    gr.Markdown("## 文字生成音频")
    system_prompt_text_input = gr.Textbox(label="系统提示词", lines=1, value="Convert the text to speech")
    gen_text_input = gr.Textbox(label="要生成的文字", lines=5)
    generate_btn = gr.Button("生成", variant="primary")

    audio_output = gr.Audio(label="生成的音频")

    generate_btn.click(
        tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            system_prompt_text_input,
            gen_text_input,
            sample_rate,
            penalty,
            temperature,
            top_k,
            top_p,
            do_sample,
            random_seed
        ],
        outputs=[audio_output],
    )


#界面

with gr.Blocks() as app:
    gr.Markdown("# LLASA 8B TTS WebUI")
    #配置
    app_setting.render() 
    #tts
    app_tts.render()


app.launch(debug=True)  # Enable debug mode for auto-reloading
