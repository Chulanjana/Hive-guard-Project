import librosa
import soundfile as sf
import pyaudio
import numpy as np
import tempfile


def record_audio_and_extract_mfcc(duration=10, sr=22050):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=sr, input=True, frames_per_buffer=CHUNK
    )

    print("Recording...")
    frames = []
    for _ in range(0, int(sr / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording complete.")

    audio_bytes = b"".join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np /= max_val  # Normalize

    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio_np, sr)
        y, _ = librosa.load(tmp.name, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

    return mfccs_mean
