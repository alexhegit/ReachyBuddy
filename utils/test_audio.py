#!/usr/bin/env python3
"""
Audio test script for sounddevice/soundfile playback.
Usage:
  python test.py --wav /tmp/tmpv_znw8gc.wav
"""

import argparse
import subprocess
import os
import sounddevice as sd
import numpy as np
import soundfile as sf


def print_devices():
    try:
        print('Default device:', sd.default.device)
        print('Devices:')
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            name = d.get('name') if isinstance(d, dict) else str(d)
            max_in = d.get('max_input_channels', '?') if isinstance(d, dict) else '?'
            max_out = d.get('max_output_channels', '?') if isinstance(d, dict) else '?'
            print(f"{i}: {name} (max_input={max_in}, max_output={max_out})")
    except Exception as e:
        print('Failed to query devices:', e)


def play_tone():
    fs = 44100
    t = np.linspace(0, 1, fs, False)
    tone_f = (0.2 * np.sin(2 * np.pi * 440 * t)).astype('float32')
    print('\nPlaying float32 tone...')
    try:
        sd.play(tone_f, fs)
        sd.wait()
        print('Played float32 tone OK')
    except Exception as e:
        print('float32 playback failed:', e)

    print('Playing int16 tone (fallback)...')
    tone_i = np.clip(tone_f * 32767, -32768, 32767).astype('int16')
    try:
        sd.play(tone_i, fs)
        sd.wait()
        print('Played int16 tone OK')
    except Exception as e:
        print('int16 playback failed:', e)


def check_wav(path):
    print(f"\nChecking WAV: {path}")
    if not os.path.exists(path):
        print('WAV not found:', path)
        return
    try:
        data, sr = sf.read(path, dtype='float32')
        print('WAV sr:', sr, 'shape:', data.shape, 'dtype:', data.dtype, 'size_bytes:', os.path.getsize(path))
    except Exception as e:
        print('Failed to read WAV via soundfile:', e)
        return

    print('Attempt float32 playback...')
    try:
        sd.play(data, samplerate=sr)
        sd.wait()
        print('float32 play OK')
        return
    except Exception as e:
        print('float32 play failed:', e)

    try:
        int16 = np.clip(data * 32767, -32768, 32767).astype('int16')
        sd.play(int16, samplerate=sr)
        sd.wait()
        print('int16 play OK')
        return
    except Exception as e:
        print('int16 play failed:', e)

    # Try system players
    for cmd in (['aplay', path], ['ffplay', '-nodisp', '-autoexit', path]):
        try:
            print('Trying system player:', cmd[0])
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"{cmd[0]} played OK")
            return
        except Exception as e:
            print(f"{cmd[0]} failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wav', default=None, help='Path to WAV file to test')
    args = p.parse_args()

    print_devices()
    play_tone()
    if args.wav:
        check_wav(args.wav)


if __name__ == '__main__':
    main()
