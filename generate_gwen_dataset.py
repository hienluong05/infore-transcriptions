#!/usr/bin/env python3
"""Generate a resumable, single-speaker Vietnamese TTS dataset with Gwen-TTS.

One nonempty input line is a source utterance; long lines can be split into chunks.
Only use reference voices that you have permission to clone.
"""
import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LOG = logging.getLogger('gwen_dataset')
MODEL_ID = 'g-group-ai-lab/gwen-tts-0.6B'
GENERATION = dict(temperature=0.3, top_k=20, top_p=0.9,
                  max_new_tokens=4096, repetition_penalty=2.0,
                  subtalker_do_sample=True, subtalker_temperature=0.1,
                  subtalker_top_k=20, subtalker_top_p=1.0)


def clean_text(s: str) -> str:
    return ' '.join(s.replace('\ufeff', '').strip().split())


def split_text(text: str, max_chars: int) -> list[str]:
    """Split on sentence endings then at whitespace; never silently truncate text."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?…。！？;；])\s+', text)
    chunks: list[str] = []
    current = ''
    for sentence in sentences:
        if not sentence:
            continue
        words = sentence.split(' ')
        for word in words:
            proposed = f'{current} {word}'.strip()
            if current and len(proposed) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = proposed
            # A single unbroken token can exceed max_chars; keep it intact.
        # End each sentence at sentence boundary when possible.
        if current and len(current) >= max_chars * 0.65:
            chunks.append(current)
            current = ''
    if current:
        chunks.append(current)
    return chunks


def load_utterances(path: Path, max_chars: int) -> list[dict[str, Any]]:
    items = []
    for line_no, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        text = clean_text(raw)
        if not text:
            continue
        for part_no, chunk in enumerate(split_text(text, max_chars), 1):
            items.append(dict(id=f'{line_no:06d}_{part_no:03d}', line=line_no,
                              part=part_no, text=chunk))
    return items


def wav_details(path: Path):
    try:
        return sf.info(str(path))
    except (OSError, RuntimeError, ValueError):
        return None


def existing_valid(wav: Path, record: Path, item: dict, target_sr: int) -> bool:
    if not wav.is_file() or not record.is_file():
        return False
    try:
        meta = json.loads(record.read_text(encoding='utf-8'))
        info = wav_details(wav)
        return bool(info and meta.get('text') == item['text'] and
                    meta.get('sample_rate') == target_sr and
                    info.samplerate == target_sr and info.channels == 1 and
                    info.frames > 0 and abs(info.duration - meta['duration']) < 0.02)
    except (ValueError, KeyError, TypeError, OSError):
        return False


def save_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def convert_audio(audio, src_sr: int, dst_sr: int):
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim != 1 or not arr.size or not np.isfinite(arr).all():
        raise ValueError('Output audio must be finite non-empty mono audio')
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError('Invalid sample rate')
    if src_sr != dst_sr:
        g = math.gcd(src_sr, dst_sr)
        arr = resample_poly(arr, dst_sr // g, src_sr // g).astype(np.float32)
    if not np.isfinite(arr).all() or np.max(np.abs(arr)) > 1.05:
        raise ValueError('Audio is invalid or clipped beyond normal PCM range')
    return np.clip(arr, -1, 1)


def write_manifests(output: Path, items: list[dict], sr: int):
    """Regenerate manifests solely from validated WAV+record pairs, atomically."""
    rows = []
    for item in items:
        wav = output / 'wavs' / f"{item['id']}.wav"
        record = output / 'records' / f"{item['id']}.json"
        if existing_valid(wav, record, item, sr):
            meta = json.loads(record.read_text(encoding='utf-8'))
            rows.append((item, wav, meta))

    files = {
        'metadata.csv': [f"wavs/{item['id']}.wav|{item['text']}\n" for item, _, _ in rows],
        'filelist.txt': [f"wavs/{item['id']}.wav|{item['text']}\n" for item, _, _ in rows],
        'manifest.jsonl': [json.dumps({'audio_filepath': str(wav.resolve()),
                                      'text': item['text'], 'duration': meta['duration']},
                                     ensure_ascii=False) + '\n' for item, wav, meta in rows],
        'details.jsonl': [json.dumps({**item, **meta, 'audio_filepath': str(wav.resolve())},
                                    ensure_ascii=False) + '\n' for item, wav, meta in rows],
    }
    for name, lines in files.items():
        target = output / name
        tmp = output / (name + '.tmp')
        with tmp.open('w', encoding='utf-8', newline='') as f:
            f.writelines(lines)
        os.replace(tmp, target)
    return len(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-file', type=Path, required=True)
    p.add_argument('--ref-audio', type=Path, required=True)
    p.add_argument('--ref-text', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sample-rate', type=int, default=22050,
                   help='Training WAV sample rate; set to match your FastPitch config (default: 22050)')
    p.add_argument('--max-chars', type=int, default=180,
                   help='Split input lines at word/sentence boundaries; 0 disables splitting')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--attention', choices=['sdpa', 'flash_attention_2'], default='sdpa',
                   help='Use flash_attention_2 only if installed and supported')
    p.add_argument('--precision', choices=['auto', 'fp16', 'bf16', 'fp32'], default='auto')
    p.add_argument('--min-seconds', type=float, default=0.3)
    p.add_argument('--max-seconds', type=float, default=40.0)
    p.add_argument('--limit', type=int, default=0, help='Process first N utterances, 0 = all')
    p.add_argument('--dry-run', action='store_true', help='Inspect inputs without loading model')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for name, path in [('Text file', args.text_file), ('Reference audio', args.ref_audio),
                       ('Reference transcript', args.ref_text)]:
        if not path.is_file():
            raise FileNotFoundError(f'{name} not found: {path}')
    if args.sample_rate <= 0 or args.max_chars < 0 or args.limit < 0:
        raise ValueError('Invalid sample-rate/max-chars/limit')
    ref_text = clean_text(args.ref_text.read_text(encoding='utf-8-sig'))
    if not ref_text:
        raise ValueError('Reference transcript is empty')
    ref_info = wav_details(args.ref_audio)
    if not ref_info or ref_info.frames == 0:
        raise ValueError('Cannot read reference WAV')
    items = load_utterances(args.text_file, args.max_chars)
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise ValueError('No nonempty utterances found')
    out = args.output_dir.resolve()
    (out / 'wavs').mkdir(parents=True, exist_ok=True)
    (out / 'records').mkdir(parents=True, exist_ok=True)
    LOG.info('Source: %s | utterances: %d | ref: %s (%.1fs) | output: %s',
             args.text_file, len(items), args.ref_audio, ref_info.duration, out)
    LOG.info('Example: %s', items[0])
    if args.dry_run:
        return

    import torch
    from qwen_tts import Qwen3TTSModel
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('PyTorch cannot access GPU; check nvidia-smi, torch CUDA build and driver')
    if args.precision == 'auto':
        dtype = (torch.bfloat16 if args.device.startswith('cuda') and
                 torch.cuda.is_bf16_supported() else
                 torch.float16 if args.device.startswith('cuda') else torch.float32)
    else:
        dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16,
                 'fp32': torch.float32}[args.precision]
    LOG.info('Loading %s on %s | dtype=%s | attention=%s',
             MODEL_ID, args.device, dtype, args.attention)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map=args.device, dtype=dtype,
        attn_implementation=args.attention)
    completed = failed = 0
    failures = out / 'failures.jsonl'
    try:
        for index, item in enumerate(items, 1):
            wav = out / 'wavs' / f"{item['id']}.wav"
            rec = out / 'records' / f"{item['id']}.json"
            if existing_valid(wav, rec, item, args.sample_rate):
                LOG.info('[%d/%d] SKIP %s', index, len(items), item['id'])
                completed += 1
                continue
            temp_wav = wav.with_suffix('.wav.tmp')
            try:
                with torch.inference_mode():
                    generated, orig_sr = model.generate_voice_clone(
                        text=item['text'], language='Vietnamese',
                        ref_audio=str(args.ref_audio.resolve()), ref_text=ref_text,
                        **GENERATION)
                audio = convert_audio(generated[0], int(orig_sr), args.sample_rate)
                duration = len(audio) / args.sample_rate
                if not args.min_seconds <= duration <= args.max_seconds:
                    raise ValueError(f'Unexpected duration: {duration:.2f}s')
                sf.write(str(temp_wav), audio, args.sample_rate, format='WAV', subtype='PCM_16')
                os.replace(temp_wav, wav)
                save_json_atomic(rec, {'text': item['text'], 'duration': duration,
                                       'sample_rate': args.sample_rate,
                                       'model': MODEL_ID, 'source_line': item['line'],
                                       'source_part': item['part']})
                completed += 1
                LOG.info('[%d/%d] OK %s %.2fs', index, len(items), wav.name, duration)
            except Exception as e:
                failed += 1
                temp_wav.unlink(missing_ok=True)
                LOG.exception('[%d/%d] FAILED %s', index, len(items), item['id'])
                with failures.open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': item['id'], 'text': item['text'],
                                        'error': str(e)}, ensure_ascii=False) + '\n')
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()
            if index % 20 == 0:
                write_manifests(out, items, args.sample_rate)
    finally:
        count = write_manifests(out, items, args.sample_rate)
        LOG.info('Saved %d valid samples to manifests | completed=%d failed=%d',
                 count, completed, failed)
    LOG.info('Dataset: %s', out)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning('Interrupted. Resume with the same command; existing WAVs are preserved.')
        sys.exit(130)
