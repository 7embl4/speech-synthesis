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
You can see demo [here](https://colab.research.google.com/drive/1oFnIV6KqEC3vzxHK2jteZ5nWr1IU4TBj?usp=sharing).

## Pretrained Model
The result model is on [HuggingFace](...) you may download it using:
```bash
huggingface-cli download artem1085715/hifigan --local-dir models
```

## Training

To train a model, run the following command:

```bash
python train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

## Synthesis

To synthesize audio from your dataset (if `text_dir` doesn't exist, it will download dataset from link in `.env` file;
the variable name is `CUSTOM_DATASET_URL`):

```bash
python synthesize.py --text_dir <PATH_TO_DIR_WITH_TEXTS>
```

Or you can generate it from command line:

```bash
python synthesize.py --text <YOUR_TEXT>
```

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
