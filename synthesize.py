import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.TTS import FastSpeech2
from tqdm import tqdm

from src.datasets import CustomUtteranceDataset
from src.metrics import NeuralMOS
from src.model import HiFiGAN

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--text_dir", type=str, help="Path to texts (text in `--text` will be ignored)"
    )
    parser.add_argument(
        "--text", type=str, help="Input text to convert it to speech from command line"
    )
    parser.add_argument(
        "--results_dir", type=str, help="Directory where results will be saved"
    )
    args = parser.parse_args()
    return args


def load_vocoder(model_path="models/hifigan.pth"):
    model = HiFiGAN(msd_hid_channels=128)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def load_encoder():
    encoder = FastSpeech2.from_hparams(
        source="speechbrain/tts-fastspeech2-ljspeech",
        savedir="models",
        run_opts={"device": device},
    )
    return encoder


def get_mos(gen_audio):
    met = NeuralMOS("models/nisqa_mos_only.tar")
    res = met(gen_audio)
    return res


def from_dir(args):
    # models
    encoder = load_encoder()
    vocoder = load_vocoder()

    # dataset
    dataset = CustomUtteranceDataset(args.text_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    # prediction
    total_mos = 0
    for idx, item in enumerate(tqdm(dataset, desc="Synthesis")):
        with torch.no_grad():
            mel_spec, durations, pitch, energy = encoder.encode_text([item["text"]])
            gen_audio = vocoder.generate(mel_spec)["gen_audio"]
            gen_audio = gen_audio.detach().cpu()

        torchaudio.save(
            Path(args.results_dir) / f"generated_{idx}.wav", gen_audio.squeeze(0), 22050
        )
        total_mos += get_mos(gen_audio)

    print(f"Mean MOS of generated audios: {total_mos / len(dataset)}")


def from_cli(args):
    # models
    encoder = load_encoder()
    vocoder = load_vocoder()

    # prediction
    mel_spec, durations, pitch, energy = encoder.encode_text([args.text])
    gen_audio = vocoder.generate(mel_spec)["gen_audio"]
    gen_audio = gen_audio.detach().cpu()

    # save audio
    os.makedirs(args.results_dir, exist_ok=True)
    time = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    torchaudio.save(
        Path(args.results_dir) / f"gen_from_cli_{time}.wav", gen_audio.squeeze(0), 22050
    )

    # calc mos
    mos = get_mos(gen_audio)
    print(f"MOS of generated audio: {mos:.4f}")  # noqa


if __name__ == "__main__":
    args = parse_args()
    if args.text_dir:
        from_dir(args)
    elif args.text:
        from_cli(args)
    else:
        raise ValueError(
            "Provide either text using --text or text directory using --text_dir"
        )
