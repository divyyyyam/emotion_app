"""
Model Shape Inspector
Run this BEFORE starting the backend to verify your model input/output shapes.
Usage: python inspect_models.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF noise

import tensorflow as tf
import numpy as np

MODELS = {
    "face":   "models/face_emotion_fer2013.keras",
    "speech": "models/speech_emotion_ravdess_cnn.keras",
    "text":   "models/text_emotion_trained.keras",
}

EXPECTED = {
    "face":   {"input": (None, 48, 48, 1),  "output": (None, 7)},
    "speech": {"input": (None, 40, 130, 1), "output": (None, 8)},
    "text":   {"input": (None, 100),         "output": (None, 7)},
}

SEP = "─" * 60

def check_shape(actual, expected, name):
    if actual == expected:
        return f"  ✓  {name}: {actual}"
    else:
        return f"  ✗  {name}: got {actual}  (expected {expected})"

def inspect(name, path):
    print(f"\n{SEP}")
    print(f"  MODEL: {name.upper()}  →  {path}")
    print(SEP)

    if not os.path.exists(path):
        print(f"  ✗  FILE NOT FOUND: {path}")
        return None

    try:
        model = tf.keras.models.load_model(path)
    except Exception as e:
        print(f"  ✗  Failed to load: {e}")
        return None

    # ── Input shape ──────────────────────────────────
    in_shape  = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)

    print(f"\n  Input  shape : {in_shape}")
    print(f"  Output shape : {out_shape}")
    print(f"  Parameters   : {model.count_params():,}")
    print(f"  Layers       : {len(model.layers)}")

    # ── Compare to expected ──────────────────────────
    exp = EXPECTED.get(name)
    if exp:
        print()
        print(check_shape(in_shape,  exp["input"],  "input "))
        print(check_shape(out_shape, exp["output"], "output"))

    # ── Layer-by-layer summary ───────────────────────
    print(f"\n  {'Layer':<30} {'Type':<22} {'Output shape'}")
    print(f"  {'─'*30} {'─'*22} {'─'*20}")
    for layer in model.layers:
        try:
            out = str(layer.output_shape)
        except Exception:
            out = "n/a"
        print(f"  {layer.name:<30} {type(layer).__name__:<22} {out}")

    # ── Quick dummy inference ────────────────────────
    print(f"\n  Running dummy inference...")
    try:
        dummy_shape = list(in_shape)
        dummy_shape[0] = 1
        # replace None dims with a safe default
        dummy_shape = [d if d is not None else 1 for d in dummy_shape]
        dummy = np.zeros(dummy_shape, dtype=np.float32)
        out = model.predict(dummy, verbose=0)
        print(f"  ✓  Inference OK  →  output shape {out.shape}, values sum to {out.sum():.4f}")
        print(f"     Sample output: {np.round(out[0], 3)}")
    except Exception as e:
        print(f"  ✗  Inference failed: {e}")

    return {"input": in_shape, "output": out_shape}


def print_fix_hints(results):
    print(f"\n{SEP}")
    print("  FIX HINTS FOR main.py")
    print(SEP)

    face = results.get("face")
    if face and face["input"] != (None, 48, 48, 1):
        s = face["input"]
        print(f"\n  Face model input is {s}")
        if len(s) == 4:
            print(f"  → Update preprocess_face() reshape to: img.reshape(1, {s[1]}, {s[2]}, {s[3]})")
            print(f"    and cv2.resize to ({s[1]}, {s[2]})")
        elif len(s) == 3:
            print(f"  → Model expects no channel dim. Use: img.reshape(1, {s[1]}, {s[2]})")

    speech = results.get("speech")
    if speech and speech["input"] != (None, 40, 130, 1):
        s = speech["input"]
        print(f"\n  Speech model input is {s}")
        if len(s) == 4:
            n_mfcc  = s[1]
            n_frames = s[2]
            print(f"  → Update preprocess_speech():")
            print(f"     n_mfcc={n_mfcc}, target_len={n_frames}")
            print(f"     mfcc.reshape(1, {n_mfcc}, {n_frames}, 1)")
        elif len(s) == 3:
            n_mfcc  = s[1]
            n_frames = s[2]
            print(f"  → Model has no channel dim. Use:")
            print(f"     mfcc.reshape(1, {n_mfcc}, {n_frames})")
        elif len(s) == 2:
            print(f"  → Model takes flat input of size {s[1]}")
            print(f"     Flatten MFCC: mfcc.flatten()[:{s[1]}].reshape(1, {s[1]})")

    text = results.get("text")
    if text and text["output"] != (None, 7):
        n = text["output"][-1]
        print(f"\n  Text model has {n} output classes (expected 7)")
        if n == 8:
            print(f"  → Already handled by normalize_ravdess_output() in main.py ✓")
        else:
            print(f"  → Add a custom class mapping in main.py for {n} classes")

    if all(
        results.get("face",   {}).get("input")  == (None, 48, 48, 1) and
        results.get("speech", {}).get("input")  == (None, 40, 130, 1) and
        results.get("text",   {}).get("output") in [(None,7),(None,8)]
        for _ in [1]
    ):
        print("\n  All shapes match expected values — no changes needed in main.py ✓")


if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print("  MULTIMODAL EMOTION MODEL INSPECTOR")
    print(f"{'═'*60}")
    print(f"  TensorFlow version: {tf.__version__}")

    results = {}
    for name, path in MODELS.items():
        r = inspect(name, path)
        if r:
            results[name] = r

    print_fix_hints(results)
    print(f"\n{'═'*60}\n")
