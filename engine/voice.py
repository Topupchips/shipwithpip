"""Offline character voices (Kokoro via sherpa-onnx). Free, no account, same voice every run."""
import os
import numpy as np
import sherpa_onnx
import soundfile as sf

# kokoro-en-v0_19 speaker ids
SPEAKERS = {"af": 0, "af_bella": 1, "af_nicole": 2, "af_sarah": 3, "af_sky": 4,
            "am_adam": 5, "am_michael": 6, "bf_emma": 7, "bf_isabella": 8, "bm_george": 9, "bm_lewis": 10}
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"

_tts = None


V1 = {"af_alloy": 0, "af_aoede": 1, "af_bella": 2, "af_heart": 3, "af_jessica": 4, "af_kore": 5, "af_nicole": 6, "af_nova": 7,
      "af_river": 8, "af_sarah": 9, "af_sky": 10, "am_adam": 11, "am_echo": 12, "am_eric": 13, "am_fenrir": 14, "am_liam": 15,
      "am_michael": 16, "am_onyx": 17, "am_puck": 18, "am_santa": 19}


def load(model_dir):
    global _tts
    if _tts is None and "multi-lang" in model_dir:
        _tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=os.path.join(model_dir, "model.onnx"), voices=os.path.join(model_dir, "voices.bin"),
                tokens=os.path.join(model_dir, "tokens.txt"), data_dir=os.path.join(model_dir, "espeak-ng-data"),
                dict_dir=os.path.join(model_dir, "dict"), lexicon=os.path.join(model_dir, "lexicon-us-en.txt"), lang="en-us"),
            num_threads=4)))
    if _tts is None:
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=os.path.join(model_dir, "model.onnx"),
                    voices=os.path.join(model_dir, "voices.bin"),
                    tokens=os.path.join(model_dir, "tokens.txt"),
                    data_dir=os.path.join(model_dir, "espeak-ng-data")),
                num_threads=4))
        _tts = sherpa_onnx.OfflineTts(cfg)
    return _tts


def say(model_dir, text, voice, speed=1.0):
    """Return (float32 mono samples, sample_rate) with silence trimmed off both ends."""
    tts = load(model_dir)
    sid = V1[voice] if "multi-lang" in model_dir else SPEAKERS[voice]
    a = tts.generate(text, sid=sid, speed=speed)
    x = np.asarray(a.samples, dtype=np.float32)
    env = np.abs(x) > 0.01
    if env.any():
        i0, i1 = np.argmax(env), len(env) - np.argmax(env[::-1])
        x = x[max(0, i0 - 400): i1 + 800]
    return x, a.sample_rate


if __name__ == "__main__":
    import sys
    x, sr = say(sys.argv[1], sys.argv[2], sys.argv[3])
    sf.write(sys.argv[4], x, sr)
