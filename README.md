# Speech Synthesis

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#pretrained-model">Pretrained Model</a> •
  <a href="#training">Training</a> •
  <a href="#synthesis">Synthesis</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an implementation of HiFi-GAN, which trained on LJSpeech-1.1 dataset.

## Installation

```bash
git clone https://github.com/7embl4/speech-synthesis.git
cd speech-synthesis
conda create -n tts python=3.12
pip install -r requirements.txt
```

## Demo
You can see demo online in [colab](https://colab.research.google.com/drive/1oFnIV6KqEC3vzxHK2jteZ5nWr1IU4TBj?usp=sharing) or [download](https://github.com/7embl4/speech-synthesis/blob/main/demo.ipynb) notebook.

## Samples
Some of samples, generated with the model.

<audio controls>
  <source src="https://raw.githubusercontent.com/7embl4/speech-synthesis/main/samples/generated_1.wav" type="audio/mpeg">
  Your browser doesn't support this element.
</audio>

<audio controls>
  <source src="https://raw.githubusercontent.com/7embl4/speech-synthesis/main/samples/generated_2.wav" type="audio/mpeg">
  Your browser doesn't support this element.
</audio>

<audio controls>
  <source src="https://raw.githubusercontent.com/7embl4/speech-synthesis/main/samples/generated_3.wav" type="audio/mpeg">
  Your browser doesn't support this element.
</audio>

## Pretrained Model
The result model is on [HuggingFace](https://huggingface.co/artem1085715/hifigan), you may download it using:
```bash
huggingface-cli download artem1085715/hifigan --local-dir models
```

## Training

To train a model with basic config, run the following command:

```bash
python train.py
```

You can check [train config](https://github.com/7embl4/speech-synthesis/blob/main/src/configs/hifigan.yaml) for parameters to adjust.

## Synthesis

To synthesize audio from **your dataset** (if `text_dir` doesn't exist, it will download dataset from link in `.env` file;
the variable name is `CUSTOM_DATASET_URL`):

```bash
python synthesize.py --text_dir <PATH_TO_DATASET>
```

`Note`. Dataset **must be** in the following format:
```bash
NameOfTheDirectoryWithUtterances
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

You can also generate audio **from command line**:

```bash
python synthesize.py --text <YOUR_TEXT>
```

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
