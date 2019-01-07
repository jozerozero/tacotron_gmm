# Tacotron-2:
原文档: [README_.md](README_.md)  

原仓库：[Tacotron-2](https://github.com/begeekmyfriend/Tacotron-2)  

此项目为 Windows（10）修正版，修正Windows中路径、创建目录等问题，并添加了测试服务(demo_server.py)。  

数据集为清华大学开源普通话语料：[data_thchs30](http://www.openslr.org/18)， [百度云](http://pan.baidu.com/s/1hqKwE00)  

# 项目结构:
	Tacotron-2
	├── datasets
	├── data_thchs30	(0)
	│   └── data
	│   └── dev
	│   └── lm_phone
	│   └── lm_word
	│   └── test
	│   └── train
 	│   └── README.TXT
	├── logs-Tacotron	(2)
	│   ├── eval_-dir
	│   │ 	├── plots
	│ 	│ 	└── wavs
	│   ├── mel-spectrograms
	│   ├── plots
	│   ├── pretrained
	│   └── wavs
	├── logs-Wavenet	(4)
	│   ├── eval-dir
	│   │ 	├── plots
	│ 	│ 	└── wavs
	│   ├── plots
	│   ├── pretrained
	│   └── wavs
	├── papers
	├── tacotron
	│   ├── models
	│   └── utils
	├── tacotron_output	(3)
	│   ├── eval
	│   ├── gta
	│   ├── logs-eval
	│   │   ├── plots
	│   │   └── wavs
	│   └── natural
	├── wavenet_output	(5)
	│   ├── plots
	│   └── wavs
	├── training_data	(1)
	│   ├── audio
	│   ├── linear
	│	└── mels
	└── wavenet_vocoder
		└── models

The previous tree shows the current state of the repository (separate training, one step at a time).

- Step **(0)**: 下载[data_thchs30](http://www.openslr.org/18)数据集([百度云](http://pan.baidu.com/s/1hqKwE00))，并将其解压至 **data_thchs30**，如上所示。(preprocess时将会处理其子文件夹 **data** 中的数据，若想更改请修改 **datasets/preprocessor.py** 或者替换文件夹 **data** 。)。
- Step **(1)**: 数据预处理。 将会生成 **training_data** 目录。
- Step **(2)**: 训练模型。其中生成的模型、对齐图等将保存至 **logs-Tacotron** 目录内。
- Step **(3)**: 生成音频。生成的音频将保存至 **tacotron_output** 目录内。（或者运行demo_server.py，指定模型并且在线输入中文和生成普通话音频，内部使用pypinyin转换汉字。）
- Step **(4)**: Train your Wavenet model. Yield the **logs-Wavenet** folder.
- Step **(5)**: Synthesize audio using the Wavenet model. Gives the **wavenet_output** folder.