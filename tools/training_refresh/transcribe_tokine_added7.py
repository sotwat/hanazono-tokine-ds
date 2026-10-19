#!/usr/bin/env python3
import json
from pathlib import Path
import torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from transcribe_samples import audio_excerpt
from prepare_tokine_added7 import SONGS,SRC,PREP,nfc
import soundfile as sf

def main():
    PREP.mkdir(parents=True,exist_ok=True)
    out=PREP/'transcripts.jsonl'
    done={(x['song'],x['start']) for x in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    processor=WhisperProcessor.from_pretrained('openai/whisper-small',local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained('openai/whisper-small',local_files_only=True).to(device).eval()
    for song in SONGS:
        wav=next(p for p in (SRC/'DS_wav').glob('*.wav') if nfc(p.stem)==song)
        duration=sf.info(wav).duration
        for start in range(0,int(duration),25):
            if (song,start) in done: continue
            length=min(29,duration-start)
            if length<3: continue
            audio=audio_excerpt(wav,start,length)
            features=processor(audio,sampling_rate=16000,return_tensors='pt').input_features.to(device)
            with torch.inference_mode():
                ids=model.generate(features,language='ja',task='transcribe',max_new_tokens=160)
            row={'song':song,'start':start,'duration':length,'text':processor.batch_decode(ids,skip_special_tokens=True)[0]}
            with out.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(song,start,flush=True)
if __name__=='__main__': main()
