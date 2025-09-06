# Furby Voice Fine-Tuning with Piper TTS: Complete Guide

This guide walks you step-by-step through how to install Piper, prepare a dataset, fine-tune a new voice model (Furby in this case), and generate speech with it. This assumes you're using **Ubuntu 22.04**, have Python 3.10 or 3.11.

---

## 1. Environment Setup

### Step 1.1 — Install Dependencies
```bash
sudo apt update && sudo apt install -y python3-dev python3-venv ffmpeg espeak-ng aplay
```

### Step 1.2 — Clone Piper
```bash
mkdir -p ~/piper_training && cd ~/piper_training

git clone https://github.com/rhasspy/piper.git
cd piper/src/python
```

### Step 1.3 — Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 1.4 — Downgrade pip for PyTorch-Lightning compatibility
```bash
pip install pip==23.2.1
```

### Step 1.5 — Install Python Dependencies
```bash
pip install --upgrade wheel setuptools
pip install -e .
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install onnx onnxsim
```

> Note: `piper_train` depends on `pytorch-lightning~=1.7.0`, which requires PyTorch < 2.0.

Install additional packages:
```bash
pip install whisper sentence-transformers faiss-cpu sounddevice numpy requests
```

---

## 2. Preparing the Dataset

### Step 2.1 — Normalize WAV Files

Install `ffmpeg-normalize`:
```bash
pip install ffmpeg-normalize
```

Then normalize audio to -16 LUFS, 22050Hz, and mono using the normalize_wav.py script.


### Step 2.2 — Generate `metadata.csv`
For single-speaker datasets, use the metadata_generator.py script


## 3. Preprocessing with Piper

```bash
python3 -m piper_train.preprocess \
  --language en-us \
  --input-dir /home/user/piper_training/piper/src/python/furby_dataset \
  --output-dir /home/user/piper_training/piper/src/python/furby_training_data \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050
```

This will generate:
- `config.json`
- `dataset.jsonl`
- `cache/` directory of `.pt` audio and spectrogram tensors

---

## 4. Required Code Modifications

### 4.1 — Fix PyTorch Lightning Scheduler Crash

Open `piper_train/vits/lightning.py` and **replace the `configure_optimizers` function** with the following:

```python
    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
        self.model_g.parameters(),
        lr=self.hparams.learning_rate,
        betas=self.hparams.betas,
        eps=self.hparams.eps,
        )
        optimizer_d = torch.optim.AdamW(
        self.model_d.parameters(),
        lr=self.hparams.learning_rate,
        betas=self.hparams.betas,
        eps=self.hparams.eps,
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=self.hparams.lr_decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=self.hparams.lr_decay)

        scheduler_g_config = {
        "scheduler": scheduler_g,
        "interval": "epoch",
        "frequency": 1,
        "name": "lr_g",
        }
        scheduler_d_config = {
        "scheduler": scheduler_d,
        "interval": "epoch",
        "frequency": 1,
        "name": "lr_d",
        }

        return [optimizer_g, optimizer_d], [scheduler_g_config, scheduler_d_config]


    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # PyTorch Lightning 1.7 requires to define how schedulers step
        scheduler.step()
```

---

## 5. Fine-Tuning the Model

Download a base checkpoint (I used Lessac Medium with decent results):
```bash
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt -O lessac_medium.ckpt
```

Then launch training:
```bash
python3 -m piper_train \
  --dataset-dir /home/user/piper_training/piper/src/python/furby_training_data \
  --accelerator gpu \
  --devices 1 \
  --batch-size 16 \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs 3164 \
  --resume_from_checkpoint /home/user/lessac_medium.ckpt \
  --checkpoint-epochs 5 \
  --precision 32
```
You will need to change the epochs arg to be +1000 whatever the epochs are for the base model checkpoint you are finetuning. In my case it was 2164.

### Monitoring Training
Launch TensorBoard:
```bash
tensorboard --logdir /home/user/piper_training/piper/src/python/furby_training_data/lightning_logs
```

---

## 6. Export to ONNX

```bash
python3 -m piper_train.export_onnx \
  /home/user/piper_training/piper/src/python/furby_training_data/lightning_logs/version_*/checkpoints/epoch=xxxx-step=xxxxxxx.ckpt \
  /home/user/furby_finetuned.onnx

cp /home/user/piper_training/piper/src/python/furby_training_data/config.json \
   /home/user/furby_finetuned.onnx.json
```

---

## 7. Inference with Piper

Install Piper CLI:
```bash
pip install piper-tts
```

Then test:
```bash
echo "Me love snacks!" | piper -m /home/user/furby_finetuned.onnx --output_file test.wav
aplay test.wav
```



## 8. Credits
- Based on [Piper TTS](https://github.com/rhasspy/piper)
- Fine-tuned using Whisper + FAISS + LLaMA + Piper stack
- Thorsten Muellers guide was invaluable source of knowledge for this part of the project (https://github.com/thorstenMueller/Thorsten-Voice/?tab=readme-ov-file#thorsten-voice-dataset-202102-neutral)

