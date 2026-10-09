#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.metadata as md, json, os, platform, sys, time, traceback
from pathlib import Path
from PIL import Image, ImageDraw

MODEL_ID = 'black-forest-labs/FLUX.2-klein-4B'
MODEL_SOURCE = 'ModelScope local weights'
PROMPT = '''Use the provided layout image as the authoritative spatial layout reference.

Generate a photorealistic garden scene that strictly follows the supplied layout image.

CRITICAL VIEWPOINT REQUIREMENT:
The image must be a strict true top-down nadir view, with the camera positioned exactly vertically above the garden and pointing straight downward. The camera optical axis must be perpendicular to the ground plane. The result must look like a directly overhead map-like aerial image, not an angled aerial photograph.

Do not use an oblique view, tilted view, three-quarter aerial view, or cinematic perspective. Do not show the horizon. Do not show visible building facades, side walls, or side faces of objects. Structures such as the greenhouse and gazebo must be seen only from above, primarily as roof/top shapes, without visible side perspectives. Benches, fountain, pond, paths, flowerbeds, and trees must all be shown from directly overhead.

Preserve the positions, shapes, sizes, footprints, and relative arrangement of all major regions and objects exactly as shown in the layout. Do not redesign the scene. Do not move, add, remove, or replace major objects. Keep the full garden fully visible within the frame.

Style and materials:
A photorealistic top-down formal garden with lush green grass, a central stone fountain, symmetrical flowerbeds filled with colorful flowers, neat stone pathways, a glass greenhouse on the north side, wooden benches, a small pond on the south side, and a gazebo, all rendered under soft natural daylight. Use realistic foliage, realistic paving materials, realistic water, and realistic greenhouse roof materials.

Layout details:
stone path region path_main at the center; stone path region path_cross at the center; flowerbed region flowerbed_left at the west side; flowerbed region flowerbed_right at the east side; flowerbed region flowerbed_bottom_left at the south-west area; flowerbed region flowerbed_bottom_right at the south-east area; pond region pond at the south side; rectangular greenhouse greenhouse at the north side; circular fountain fountain at the center; rectangular bench bench_left at the west side; rectangular bench bench_right at the east side; circular gazebo gazebo at the south side; 8 tree elements distributed within the west side area; 8 tree elements distributed within the east side area; 6 shrub elements distributed within the north side area; 12 flower cluster elements distributed within the west side area; 12 flower cluster elements distributed within the east side area; 10 flower cluster elements distributed within the south-west area; 10 flower cluster elements distributed within the south-east area.

Keep clear visual boundaries between lawn, paths, planting beds, water, and landmarks. The result must remain structurally faithful to the layout while appearing realistic and detailed.'''

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); ap.add_argument('--model-dir',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3]); a=ap.parse_args()
    source=Path(a.source).resolve(); out=Path(a.out).resolve(); model=Path(a.model_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    (out/'prompt.txt').write_text(PROMPT,encoding='utf-8')
    meta={'FINAL_STATUS':'FAIL','MODEL_ID':MODEL_ID,'MODEL_SOURCE':MODEL_SOURCE,'MODEL_PATH':str(model),'INPUT_IMAGE':str(source),'INPUT_SHA256':sha(source),'PROMPT_SHA256':hashlib.sha256(PROMPT.encode()).hexdigest(),'PROMPT':PROMPT,'NUM_INFERENCE_STEPS':4,'SEEDS_REQUESTED':a.seeds,'SEEDS_COMPLETED':[],'OUTPUTS':[],'MULTI_REFERENCE':False,'environment':{'python':sys.version,'platform':platform.platform(),'torch':None,'diffsynth':None},'per_seed':[]}
    try:
        if not source.is_file(): raise FileNotFoundError(source)
        required=[model/'text_encoder/model-00001-of-00002.safetensors',model/'text_encoder/model-00002-of-00002.safetensors',model/'transformer/diffusion_pytorch_model.safetensors',model/'vae/diffusion_pytorch_model.safetensors',model/'tokenizer/tokenizer.json']
        missing=[str(p) for p in required if not p.is_file()]
        if missing: raise RuntimeError('ModelScope weights incomplete: '+', '.join(missing))
        import torch
        meta['environment']['torch']=torch.__version__; meta['environment']['torch_cuda']=torch.version.cuda
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; fail closed')
        meta['GPU']={'visible':os.getenv('CUDA_VISIBLE_DEVICES'),'name':torch.cuda.get_device_name(0),'total_memory_gib':round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)}
        from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
        try: meta['environment']['diffsynth']=md.version('diffsynth')
        except Exception: meta['environment']['diffsynth']='local DiffSynth-Studio'
        image=Image.open(source).convert('RGB'); w=image.width-image.width%16; h=image.height-image.height%16
        if (w,h)!=(image.width,image.height): image=image.crop((0,0,w,h))
        shards=sorted(str(p) for p in (model/'text_encoder').glob('model-*.safetensors'))
        if len(shards)!=2: raise RuntimeError(f'Expected 2 text encoder shards, found {len(shards)}')
        configs=[ModelConfig(path=shards),ModelConfig(path=str(model/'transformer/diffusion_pytorch_model.safetensors')),ModelConfig(path=str(model/'vae/diffusion_pytorch_model.safetensors'))]
        pipe=Flux2ImagePipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda',model_configs=configs,tokenizer_config=ModelConfig(path=str(model/'tokenizer')))
        for seed in a.seeds:
            torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
            im=pipe(PROMPT,edit_image=image,edit_image_auto_resize=True,height=h,width=w,seed=seed,rand_device='cuda',num_inference_steps=4)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-t
            if not isinstance(im,Image.Image) or im.size!=(w,h): raise RuntimeError(f'bad output for seed {seed}: {getattr(im,"size",None)}')
            p=out/f'seed{seed:02d}.png'; im.save(p)
            rec={'seed':seed,'path':str(p),'sha256':sha(p),'size':[w,h],'runtime_sec':round(elapsed,3),'peak_memory_gib':round(torch.cuda.max_memory_allocated()/1024**3,3)}
            meta['per_seed'].append(rec); meta['SEEDS_COMPLETED'].append(seed); meta['OUTPUTS'].append(str(p)); print(json.dumps(rec),flush=True)
        thumb_w,thumb_h=420,420; gap=16; label_h=28
        sheet=Image.new('RGB',(2*thumb_w+3*gap,2*(thumb_h+label_h)+3*gap),(30,30,30)); d=ImageDraw.Draw(sheet)
        for i,rec in enumerate(meta['per_seed']):
            im=Image.open(rec['path']).convert('RGB'); im.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS); x=gap+(i%2)*(thumb_w+gap); y=gap+(i//2)*(thumb_h+label_h+gap); sheet.paste(im,(x+(thumb_w-im.width)//2,y+(thumb_h-im.height)//2)); d.text((x,y+thumb_h+3),f"seed {rec['seed']}",fill='white')
        sheet.save(out/'contact_sheet.png'); meta['CONTACT_SHEET']=str(out/'contact_sheet.png'); meta['FINAL_STATUS']='PASS'
    except Exception as e:
        meta['ERROR']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}; print(meta['ERROR']['traceback'],file=sys.stderr)
    meta['COMPLETED_UNIX']=time.time(); (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'FINAL_STATUS':meta['FINAL_STATUS'],'seeds':meta['SEEDS_COMPLETED'],'outputs':meta['OUTPUTS'],'error':meta.get('ERROR')},ensure_ascii=False))
    return 0 if meta['FINAL_STATUS']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
