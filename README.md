# Llasa TTS 8B WebUI Demo

## 介绍

Llasa-8B是一个基于Xcodec2的语音编码器，支持8kHz采样率的音频编码。本项目提供了一个简单的Demo，用于展示Llasa-8B的编码和解码效果。同时，我们还提供了一个WebUI，用于在线体验Llasa-8B的编码和解码效果。

我用faster-whisper 替换了默认的whisper，这样速度会块很多。官方的代码用的whisper-large-v3-turbo，我替换为faster-whisper-large-v3，这样准确率也会有提升。

faster-whisper只负责将语音转换为文字，不负责编码解码，所以不是必须，只要在文本框中确认与输入的音频文本一致即可（手动输入）。

我只测了中文和英文（混合也可以），其他语言可能会有问题，如果有问题请在本页最下方找到官方的链接，提issue。


## 安装

### 依赖

```
conda create -n xcodec2 python=3.9
conda activate xcodec2
pip install -r requirements.txt
```

说明：windows下必须要使用Python 3.9,具体问题可以参考下面这个讨论：
https://huggingface.co/HKUSTAudio/xcodec2/discussions/4

我用ubuntu测试通过了，windows没测试。


### 运行

```
gradio app.py
#或 python app.py
```

## 一些说明

1、Llasa-8B 下载需要hf的token，所以需要登录hf，然后在环境变量中设置HF_TOKEN，或者先把模型下到本地，然后加载本地模型，比如：

```python
llasa_8b ='/nvme/llm/Llasa-8B'
```
这样就可以直接加载了

2、内存占用：
- Llasa-8B需要大概17G内存（我用的fp16加载）
- faster-whisper-large-v3需要大概3G，我为省一些内存把他加载到了cpu上了，速度会慢一些，但是不影响使用，如果你内存大，加遭到gpu上会更快。
- xcodec2需要大概3G内存

所以模型总计20G显存，再加上推理的中间变量24G内存是够用的，如果内存不够，可以考虑把3B的小模型，这样10G左右应该够了


3、量化：
官方的模型可以自动用fp16加载，我测试了一下量化到fp8，但是有一些问题，所以没有加入到代码中，如果你有兴趣可以试试。
如果fp8（或int8）的话，12G显存应该可以跑起来。

4、推理参数：
官方没给全部列表，我配置的是一些简单的通用推理参数，官方说采样 16000，我自己测试24000也可以，但是超过30000就不行了，所以我设置了个列表，可以自己选择采样率。


## 链接
模型：
https://huggingface.co/HKUSTAudio/Llasa-8B

官方给的微调代码：
https://github.com/zhenye234/LLaSA_training/tree/main/finetune


## license
只有本代码也就是app.py是 MIT，其他的模型和代码请参考其官方的license
