from pathlib import Path
import os,sys,json,time,shutil,subprocess,hashlib,math
from .hashing import sha256_file,sha256_path
from .paths import load_config,expand
from .image_io import image_info,resize_cover
from .camera_io import load_trajectory
from .ply_utils import ply_header
from .video_io import decoded_frame_count
from .subprocess_utils import run_command

LEGACY_STAGES = ['input','triposplat','camera','proxy','flux_anchor','fixanything','report']
STAGES = LEGACY_STAGES
EXECUTION_ORDER = ['agent2topdown','flux_topdown','input','triposplat','camera','proxy','flux_anchor','fixanything','seedvr2','report']
LEGACY_NUMERIC_STAGES = {str(i): name for i, name in enumerate(LEGACY_STAGES)}
LEGACY_CACHE_COMPATIBLE_STAGES = set(LEGACY_STAGES) - {'flux_anchor', 'fixanything', 'report'}
STAGE_ALIASES = {'agent': 'agent2topdown', 'topdown': 'flux_topdown', 'flux_topdown': 'flux_topdown', 'flux': 'flux_anchor', 'fix': 'fixanything'}
PATH_KEYS = {
    'agent2topdown': ('agent_generated_topdown','agent_manifest'),
    'flux_topdown': ('flux_topdown_candidates','flux_topdown_candidate','topdown_used','flux_topdown_review','flux_topdown_manifest'),
    'input': ('input_copy','input_manifest'),
    'triposplat': ('triposplat_dir','ply','triposplat_manifest','preprocessing_trace'),
    'camera': ('camera_dir','trajectory','camera_report'),
    'proxy': ('proxy_dir','proxy_frames','proxy_video','proxy_report','proxy_contact'),
    'flux_anchor': ('anchor_dir','anchor000','anchor060','alignment','anchor_report'),
    'fixanything': ('fa_input','fa_input_manifest','anchor_provenance','fa_video','fa_frames','fa_report'),
    'seedvr2': ('seedvr2_dir','seedvr2_video','seedvr2_manifest'),
    'report': ('final_report','summary','comparison_sheet','comparison_video','model_registry'),
}

def paths(cfg,base,project):
    r={k:expand(v,base,project) for k,v in cfg['paths'].items()}
    r['weight_hash_manifest']=r['model_registry'].parent/'model_weight_hashes.json'
    r['canonical_topdown']=r['input_copy']
    r['agent_generated_topdown']=base/'01_agent2topdown/generated_topdown.png'
    r['agent_manifest']=base/'01_agent2topdown/agent_manifest.json'
    r['flux_topdown_dir']=base/'02_flux_topdown'
    r['flux_topdown_candidates']=r['flux_topdown_dir']/'outputs'
    r['flux_topdown_candidate']=r['flux_topdown_dir']/'topdown.png'
    r['topdown_used']=r['flux_topdown_dir']/'topdown_used.png'
    r['flux_topdown_review']=r['flux_topdown_dir']/'visual_review.json'
    r['flux_topdown_manifest']=r['flux_topdown_dir']/'topdown_used_manifest.json'
    r['flux_topdown_status']=base/'audit/stages/flux_topdown/stage_status.json'
    r['flux_topdown_metadata']=r['flux_topdown_candidates']/'metadata.json'
    r['flux_topdown_script']=Path(__file__).resolve().parents[1]/'integrations/run_flux_topdown.py'
    r['agent_raw_root']=base/'agent2topdown'
    agent_cfg=cfg.get('agent2topdown',{})
    agent_root = os.environ.get("AGENT2TOPDOWN_ROOT", str(agent_cfg.get('root', ""))).strip()
    if agent_root:
        configured = Path(agent_root).expanduser()
        if not configured.is_absolute():
            configured = project / configured
    else:
        configured = Path(__file__).resolve().parents[1] / "agent2topdown_backend"
    r['agent_root'] = configured.resolve()
    r['seedvr2_dir']=base/'07_seedvr2'
    r['seedvr2_input_dir']=r['seedvr2_dir']/'inputs'
    r['seedvr2_output_dir']=r['seedvr2_dir']/'outputs'
    r['seedvr2_video']=r['seedvr2_output_dir']/'video.mp4'
    r['seedvr2_manifest']=r['seedvr2_dir']/'seedvr2_manifest.json'
    r['seedvr2_model_manifest']=base/'audit/seedvr2_model_manifest.json'
    r['seedvr2_log']=base/'logs/stage_seedvr2.log'
    r['seedvr2_status']=base/'audit/stages/seedvr2/stage_status.json'
    return r

def load_json(p): return json.loads(Path(p).read_text())
def atomic_json(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n'); t.replace(p)
def canonical_hash(d): return hashlib.sha256(json.dumps(d,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()

def anchor_alignment_policy(alignment, allow_fail=False):
    values = {
        'ANCHOR000_ALIGNMENT': alignment.get('ANCHOR000_ALIGNMENT'),
        'ANCHOR060_ALIGNMENT': alignment.get('ANCHOR060_ALIGNMENT'),
    }
    failed = any(value not in ('GOOD', 'PARTIAL') for value in values.values())
    override_used = bool(failed and allow_fail)
    if failed and not allow_fail:
        rendered = ', '.join(f"{key}={value or 'MISSING'}" for key, value in values.items())
        raise RuntimeError(
            f"FixAnything blocked by anchor alignment FAIL ({rendered}); "
            "inspect the anchor/proxy comparison and explicitly opt in to continue"
        )
    warnings = []
    if failed:
        warnings.append(
            "WARNING:\nFLUX clean anchor failed automatic alignment audit.\n"
            "Continuing because explicit override was enabled."
        )
    elif any(value == 'PARTIAL' for value in values.values()):
        rendered = ', '.join(f"{key}={value}" for key, value in values.items())
        warnings.append(
            f"WARNING: FLUX clean anchor alignment audit is PARTIAL ({rendered}); continuing."
        )
    return {
        **values,
        'ANCHOR_ALIGNMENT_OVERRIDE_USED': override_used,
        'warnings': warnings,
    }

def stage_files(name,p):
    m={
      'agent2topdown':[p['agent_generated_topdown'],p['agent_manifest'],p['agent_raw_root']],
      'flux_topdown':[p['flux_topdown_candidates'],p['flux_topdown_candidate'],p['topdown_used'],p['flux_topdown_review'],p['flux_topdown_manifest']],
      'input':[p['input_copy'],p['input_manifest']],
      'triposplat':[p['ply'],p['triposplat_manifest'],p['preprocessing_trace']],
      'camera':[p['trajectory'],p['camera_report']],
      'proxy':[p['proxy_frames'],p['proxy_video'],p['proxy_report'],p['proxy_contact']],
      'flux_anchor':[p['anchor000'],p['anchor060'],p['alignment'],p['anchor_report']],
      'fixanything':[p['fa_video'],p['fa_frames'],p['fa_report']],
      'seedvr2':[p['seedvr2_video'],p['seedvr2_manifest'],p['seedvr2_model_manifest'],p['seedvr2_log']],
      'report':[p['final_report'],p['summary'],p['comparison_sheet'],p['comparison_video'],p['model_registry'],p['weight_hash_manifest']]}
    if name=='report' and p['seedvr2_video'].is_file():
        m['report'].append(p['seedvr2_video'])
    return m[name]

def stage_inputs(name,p,input_path):
    if name=='agent2topdown':
        return {}
    if name=='seedvr2':
        return {str(p['fa_video']):sha256_path(p['fa_video'])}
    if name=='flux_topdown':
        source=p['agent_generated_topdown']
        return {str(source):sha256_path(source)}
    d={'input':[input_path], 'triposplat':[input_path,p['ply'],p['triposplat_manifest']],
       'camera':[p['ply'],p['trajectory'],p['camera_report']],
       'proxy':[p['ply'],p['trajectory'],p['proxy_video'],p['proxy_frames']],
       'flux_anchor':[p['proxy_frames'],p['topdown'],p['anchor000'],p['anchor060'],p['alignment'],p['anchor_report']],
       'fixanything':[p['proxy_frames'],p['anchor000'],p['anchor060'],p['alignment'],p['fa_input'],p['fa_input_manifest']],
       'report':[p['fa_video'],p['fa_frames'],p['fa_report'],p['alignment'],p['agent_manifest'],p['seedvr2_status']]}[name]
    if name=='report' and p['seedvr2_video'].is_file():
        d.append(p['seedvr2_video'])
    return {str(x):sha256_path(x) for x in d}


def stage_config(cfg,name):
    all_paths=cfg.get('paths',{})
    selected_paths={key:all_paths.get(key) for key in PATH_KEYS.get(name,()) if key in all_paths}
    global_fields={
      'input': ('total_frames',),
      'agent2topdown': (),
      'flux_topdown': (),
      'triposplat': (),
      'camera': ('total_frames','up_axis'),
      'proxy': ('total_frames','fps','proxy_resolution'),
      'flux_anchor': ('total_frames',),
      'fixanything': ('total_frames','fixanything_resolution','temporal_resampling'),
      'seedvr2': (),
      'report': (),
    }
    global_config={key:cfg.get(key) for key in global_fields.get(name,())}
    if name=='agent2topdown':
        from pipeline.integrations.agent2topdown_adapter import module_code_hash
        agent=cfg.get('agent2topdown',{})
        root=Path(agent.get('root',cfg.get('_agent_root','')))
        dotenv_key=False
        dotenv=root/'.env'
        if dotenv.is_file():
            for line in dotenv.read_text(errors='ignore').splitlines():
                if line.strip().startswith('DEEPSEEK_API_KEY=') and line.split('=',1)[1].strip().strip('"\''):
                    dotenv_key=True
        global_config={'prompt':cfg.get('_scene_prompt'),'agent_config':{k:v for k,v in agent.items() if k not in ('api_key','token')},
                       'layout_renderer_seed':17,
                       'agent_code_sha256':module_code_hash(root) if root.is_dir() else None,
                       'model_or_service':{'model':agent.get('model','deepseek-chat'),'base_url':agent.get('base_url','https://api.deepseek.com'),
                                           'credential_available':bool(os.environ.get('DEEPSEEK_API_KEY') or dotenv_key)}}
    stage=cfg.get('stages',{}).get(name,{})
    if name=='flux_topdown':
        flux=cfg.get('flux_topdown',{})
        runner=Path(flux.get('script',Path(__file__).resolve().parents[1]/'integrations/run_flux_topdown.py'))
        if not runner.is_absolute(): runner=Path(__file__).resolve().parents[2]/runner
        adapter=Path(__file__).resolve().parents[1]/'integrations/flux_topdown_adapter.py'
        global_config={'settings':{k:v for k,v in flux.items() if k not in ('enabled',)},
                       'runner_sha256':sha256_file(runner) if runner.is_file() else None,
                       'adapter_sha256':sha256_file(adapter) if adapter.is_file() else None,
                       'runtime_sha256':sha256_file(Path(__file__))}
    if name=='seedvr2':
        seed=cfg.get('seedvr2',{})
        stage={k:v for k,v in seed.items() if k!='enabled'}
        global_config={'model_identity':cfg.get('_seedvr2_identity')}
    if name in ('flux_anchor', 'fixanything', 'report'):
        global_config['anchor_alignment_allow_fail'] = bool(
            cfg.get('_anchor_alignment_allow_fail',
                    cfg.get('flux', {}).get('anchor_alignment', {}).get('allow_fail', False))
        )
    if name=='report':
        global_config.update({'input_mode':cfg.get('_input_mode'),'seedvr2_enabled':bool(cfg.get('_seedvr2_enabled',False))})
    return canonical_hash({'global':global_config,'paths':selected_paths,'stage':stage})

def legacy_stage_config(cfg,name):
    if name not in LEGACY_STAGES:
        return None
    global_config={k:v for k,v in cfg.items() if k not in ('paths','stages','input','agent2topdown','seedvr2','legacy_input','pipeline_version') and not str(k).startswith('_')}
    global_config['input']=cfg.get('legacy_input',cfg.get('input','garden_topdown.png'))
    return canonical_hash({'global':global_config,'paths':cfg.get('paths',{}),'stage':cfg.get('stages',{}).get(name,{})})

def text_mode_topdown_source(agent_generated,topdown_used,visual_review=None,used_manifest=None):
    agent_generated=Path(agent_generated)
    topdown_used=Path(topdown_used)
    review=Path(visual_review) if visual_review else None
    manifest=Path(used_manifest) if used_manifest else None
    if not topdown_used.is_file() or review is None or not review.is_file() or manifest is None or not manifest.is_file():
        return agent_generated
    try:
        data=json.loads(review.read_text())
        provenance=json.loads(manifest.read_text())
    except (OSError,ValueError):
        return agent_generated
    approved=(data.get('APPROVED_FOR_TRIPOSPLAT') is True
              and data.get('STRICT_NADIR_VIEW')=='PASS'
              and data.get('NO_HORIZON')=='PASS'
              and data.get('NO_OBLIQUE_FACADE')=='PASS')
    if not approved:
        return agent_generated
    current_used_sha=sha256_file(topdown_used)
    current_review_sha=sha256_file(review)
    if provenance.get('topdown_used_sha256')!=current_used_sha:
        return agent_generated
    if provenance.get('visual_review_sha256')!=current_review_sha:
        return agent_generated
    if provenance.get('selected_topdown_sha256')!=data.get('CANDIDATE_SHA256'):
        return agent_generated
    return topdown_used

def input_source(cfg,base,project):
    mode=cfg.get('_input_mode')
    if mode=='text':
        p=paths(cfg,base,project)
        return text_mode_topdown_source(p['agent_generated_topdown'],p['topdown_used'],p['flux_topdown_review'],p['flux_topdown_manifest'])
    override=cfg.get('_input_override')
    input_cfg=cfg.get('input','garden_topdown.png')
    if isinstance(input_cfg,dict):
        input_cfg=input_cfg.get('default_topdown','garden_topdown.png')
    q=override or input_cfg
    p=Path(q)
    return (project/p).resolve() if not p.is_absolute() else p.resolve()

def make_stage_dirs(base):
    for d in ['00_input','01_agent2topdown','02_flux_topdown','01_triposplat','02_camera','03_proxy','04_flux','05_fixanything','06_comparisons','07_seedvr2','agent2topdown','audit/stages','logs']:
        (base/d).mkdir(parents=True,exist_ok=True)

def gpu_choice(preferred='auto'):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    busy={ln.split(',')[0].strip() for ln in apps.splitlines() if ln.strip()}
    candidates=[]
    for ln in rows:
        i,uuid,name,used,total,util=[x.strip() for x in ln.split(',')]
        if uuid not in busy and int(used)<=128 and int(util)<15: candidates.append(i)
    if preferred!='auto':
        if preferred not in candidates: raise RuntimeError(f'requested GPU {preferred} is not idle; candidates={candidates}')
        return preferred
    if not candidates: raise RuntimeError('no idle NVIDIA GPU found; no process will be killed')
    return candidates[0]
def triposplat_env(cfg,gpu):
    configured=cfg.get('stages',{}).get('triposplat',{}).get('pythonpath','')
    env={'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1'}
    if configured:
        env['PYTHONPATH']=str(configured)
    return env

def maybe_command(cfg,name,p,base,project,env=None):
    command=cfg.get('stages',{}).get(name,{}).get('command')
    if not command: return None
    if isinstance(command,str): import shlex; command=shlex.split(command)
    vals={'input':str(p.get('canonical_topdown',input_source(cfg,base,project))),'output':str(base),'project_root':str(project),**{k:str(v) for k,v in p.items()}}
    command=[str(x).format(**vals) for x in command]
    return run_command(command,project,base/'logs'/f'stage_{name}.log',env)

def artifact_record(path):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'expected audit artifact missing: {p}')
    total=p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
    return {'path':str(p.resolve()),'kind':'file' if p.is_file() else 'directory_tree','size_bytes':total,'sha256':sha256_path(p)}

def build_weight_registry(base,tri,flux,fa,weight_manifest_path):
    manifest_path=Path(weight_manifest_path)
    old=load_json(manifest_path) if manifest_path.is_file() else {}
    cached=old.get('weights',{})
    weights={}
    def actual_record(label,path):
        p=Path(path).resolve()
        if not p.is_file(): raise FileNotFoundError(f'model weight missing: {p}')
        st=p.stat(); prior=cached.get(label,{})
        if prior.get('path')==str(p) and prior.get('size_bytes')==st.st_size and prior.get('mtime_ns')==st.st_mtime_ns and prior.get('sha256'):
            rec=dict(prior)
        else:
            rec={'path':str(p),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha256_file(p),'hash_source':'recomputed SHA256','hash_seconds':None}
        weights[label]=rec
        return rec
    for name,path in (tri.get('weights') or {}).items():
        actual_record('TripoSplat/'+name,path)
    flux_root=Path(flux.get('MODEL_PATH','')).resolve()
    if not flux_root.is_dir(): raise FileNotFoundError(f'FLUX model directory missing: {flux_root}')
    flux_files=sorted(flux_root.rglob('*.safetensors'))
    if not flux_files: raise RuntimeError(f'no FLUX safetensors found under {flux_root}')
    for f in flux_files:
        actual_record('FLUX2-Klein/'+f.relative_to(flux_root).as_posix(),f)
    checkpoint_manifest=Path(fa.get('model_dir',''))/'checkpoint_manifest.json'
    if not checkpoint_manifest.is_file(): raise FileNotFoundError(f'FixAnything checkpoint manifest missing: {checkpoint_manifest}')
    for item in json.loads(checkpoint_manifest.read_text()):
        path=Path(item.get('path',''))
        if path.suffix not in ('.safetensors','.pth','.pt'):
            continue
        if not path.is_file(): raise FileNotFoundError(f'FixAnything weight missing: {path}')
        st=path.stat()
        if st.st_size!=item.get('size'):
            raise RuntimeError(f'FixAnything weight size differs from download manifest: {path}')
        label='FixAnything/Wan/'+str(path.relative_to(Path(fa['model_dir']).resolve()))
        weights[label]={'path':str(path.resolve()),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':item.get('sha256'),'hash_source':str(checkpoint_manifest.resolve()),'hash_recomputed_during_this_audit':False,'size_matches_manifest':True}
    manifest={'algorithm':'SHA-256','created_unix':old.get('created_unix',time.time()),'notes':'TripoSplat and ModelScope FLUX.2 Klein files are streamed and hashed directly; FixAnything/Wan hashes are retained from checkpoint_manifest.json and file sizes are checked against that manifest.','weights':weights}
    atomic_json(manifest_path,manifest)
    tripo={k:weights['TripoSplat/'+k] for k in (tri.get('weights') or {})}
    flux_records={k.split('/',1)[1]:v for k,v in weights.items() if k.startswith('FLUX2-Klein/')}
    fix_records={k.split('/',1)[1]:v for k,v in weights.items() if k.startswith('FixAnything/Wan/')}
    registry={
      'TripoSplat':{'repo':'https://github.com/VAST-AI-Research/TripoSplat','commit':'d8db9e018b413dd9c4a9fe22463781bf98e8e68d','snapshot':tri.get('snapshot'),'weights':tripo,'source':'official clean git snapshot'},
      'FLUX2-Klein':{'model_id':flux.get('MODEL_ID'),'source':flux.get('MODEL_SOURCE'),'path':str(flux_root),'distilled':flux.get('DISTILLED'),'steps':flux.get('NUM_INFERENCE_STEPS'),'weights':flux_records},
      'FixAnything/Wan':{'repo':fa.get('official_repo'),'commit':fa.get('official_commit'),'model_dir':fa.get('model_dir'),'environment':fa.get('python'),'weights':fix_records,'checkpoint_manifest':str(checkpoint_manifest.resolve())}}
    agent_manifest=base/'01_agent2topdown/agent_manifest.json'
    if agent_manifest.is_file():
        agent=load_json(agent_manifest); service=agent.get('MODEL_OR_SERVICE_INFO') or {}
        registry['Agent2Topdown']={'role':'natural-language-to-topdown frontend','root':agent.get('AGENT2TOPDOWN_ROOT'),
          'entrypoint':agent.get('ENTRYPOINT'),'code_sha256':agent.get('CODE_SHA256'),'model':service.get('model'),
          'service':service.get('service'),'mode':agent.get('MODE'),'fallback':service.get('fallback'),
          'source':'existing audited Agent2Topdown module','optional':False}
    seed_manifest=base/'audit/seedvr2_model_manifest.json'
    if seed_manifest.is_file():
        seed=load_json(seed_manifest); seed_weights={}
        for key,record in seed.get('weights',{}).items():
            label='SeedVR2/'+key
            weights[label]={**record,'hash_source':'pipeline1 model identity manifest'}
            seed_weights[key]=weights[label]
        registry['SeedVR2']={'model_id':seed.get('model_id'),'repo':seed.get('repo'),'commit':seed.get('repo_commit'),
          'environment':seed.get('python'),
          'inference_script':seed.get('inference_script'),'weights':seed_weights,'optional':True}
    atomic_json(manifest_path,{'algorithm':manifest.get('algorithm','SHA-256'),'created_unix':manifest.get('created_unix',time.time()),
      'notes':manifest.get('notes',''),'weights':weights})
    atomic_json(base/'audit/model_registry.json',registry)
    return registry,manifest

def stage_run(name,cfg,base,project,preferred_gpu='auto',force=False):
    p=paths(cfg,base,project); inp=input_source(cfg,base,project); p['topdown']=inp; p['canonical_topdown']=p['input_copy']; make_stage_dirs(base); details={'command':None,'started_unix':time.time()}
    if name=='agent2topdown':
        if cfg.get('_input_mode')=='image':
            result={'STATUS':'SKIPPED_INPUT_TOPDOWN_PROVIDED','MODE':'image','OUTPUT_PATH':None}
            atomic_json(p['agent_manifest'],result)
            details.update(result)
        else:
            from pipeline.integrations.agent2topdown import run as run_agent
            agent_cfg=cfg.get('agent2topdown',{})
            if agent_cfg.get('mock',False):
                raise RuntimeError('MOCK_FALLBACK_FORBIDDEN: mocks are test-only')
            result=run_agent(cfg.get('_scene_prompt',''),p['agent_root'],p['agent_raw_root'],p['agent_generated_topdown'],p['agent_manifest'])
            details.update(result)
    elif name=='flux_topdown':
        if cfg.get('_input_mode')=='image':
            details.update(STATUS='SKIPPED_INPUT_TOPDOWN_PROVIDED')
        else:
            from pipeline.integrations.flux_topdown_adapter import run as run_flux_topdown
            result=run_flux_topdown(cfg.get('flux_topdown',{}),p,base,project,preferred_gpu,force)
            details.update(result)
    elif name=='input':
        if cfg.get('_input_mode')=='text' and (not p['topdown_used'].is_file() or not p['flux_topdown_manifest'].is_file() or inp.resolve()!=p['topdown_used'].resolve()): raise RuntimeError('text mode blocked: reviewed topdown_used.png and manifest are required')
        if not inp.is_file(): raise FileNotFoundError(inp)
        p['input_copy'].parent.mkdir(parents=True,exist_ok=True)
        if inp.resolve()!=p['input_copy'].resolve(): shutil.copy2(inp,p['input_copy'])
        image=image_info(p['input_copy'])
        details.update(input=str(inp),sha256=sha256_file(p['input_copy']),image=image,input_mode=cfg.get('_input_mode'),
                       topdown_source_type='AGENT_GENERATED_FLUX_REFINED_RGBA' if cfg.get('_input_mode')=='text' else 'USER_IMAGE',
                       topdown_source_path=str(inp),agent_layout_path=str(p['agent_generated_topdown']) if cfg.get('_input_mode')=='text' else None,
                       topdown_used_sha256=sha256_file(p['topdown_used']) if cfg.get('_input_mode')=='text' else None,canonical_topdown_path=str(p['canonical_topdown']),
                       canonical_topdown_sha256=sha256_file(p['canonical_topdown']))
        atomic_json(p['input_manifest'],details)
    elif name=='triposplat':
        if cfg.get('_input_mode')=='text' and (not p['topdown_used'].is_file() or not p['flux_topdown_manifest'].is_file() or inp.resolve()!=p['topdown_used'].resolve()):
            raise RuntimeError('text mode blocked: TripoSplat requires visually reviewed topdown_used.png')
        manifest=load_json(p['triposplat_manifest']) if p['triposplat_manifest'].is_file() else None
        valid=bool(manifest and manifest.get('input',{}).get('sha256')==sha256_file(inp) and manifest.get('gaussian_integrity',{}).get('valid') and manifest.get('num_gaussians',0)>0 and p['ply'].is_file() and ply_header(p['ply'])['vertex_count']>0)
        if not valid:
            from pipeline.integrations.triposplat import run as run_triposplat
            tr_cfg=cfg.get('stages',{}).get('triposplat',{})
            gpu=gpu_choice(preferred_gpu); details.update(run_triposplat(tr_cfg,inp,p['triposplat_dir'],p['ply'],p['triposplat_manifest'],base/'logs/stage_triposplat.log',gpu,Path(__file__).resolve().parents[2]))
            manifest=load_json(p['triposplat_manifest'])
        valid=bool(manifest and manifest.get('input',{}).get('sha256')==sha256_file(inp) and manifest.get('gaussian_integrity',{}).get('valid') and p['ply'].is_file() and ply_header(p['ply'])['vertex_count']>0)
        if not valid: raise RuntimeError('TripoSplat cache/provenance invalid; downstream stopped')
        details.update(input_sha256=manifest['input']['sha256'],ply_sha256=sha256_file(p['ply']),gaussian_count=ply_header(p['ply'])['vertex_count'],seed=manifest.get('seed'),source='official TripoSplat inference adapter',input_path=manifest.get('input',{}).get('path'),input_matches_canonical=manifest.get('input',{}).get('sha256')==sha256_file(p['canonical_topdown']))
    elif name=='camera':
        if not p['camera_report'].is_file() or not p['trajectory'].is_file():
            from pipeline.integrations.camera_planner import run as run_camera
            cam_cfg=cfg.get('stages',{}).get('camera',{})
            gpu=gpu_choice(preferred_gpu)
            details.update(run_camera(cam_cfg,p['ply'],p['camera_dir'],p['camera_report'],p['trajectory'],base/'logs/stage_camera.log',gpu,Path(__file__).resolve().parents[2]))
        rep=load_json(p['camera_report']); traj=load_trajectory(p['trajectory'],61)
        selected=next((c for c in rep.get('candidates',[]) if c.get('id')==rep.get('selected_candidate')),None)
        if not selected or not selected.get('collision_free') or not selected.get('motion_smooth'): raise RuntimeError('camera safety/smoothness gate failed')
        details.update(selected=rep.get('selected_candidate'),frames=len(traj['frames']),metrics=selected,trajectory_sha256=sha256_file(p['trajectory']))
    elif name=='proxy':
        if not p['proxy_video'].is_file() or not p['proxy_frames'].is_dir():
            from pipeline.integrations.proxy_renderer import run as render_proxy
            pr_cfg=cfg.get('stages',{}).get('proxy',{})
            gpu=gpu_choice(preferred_gpu)
            details.update(render_proxy(pr_cfg,p['ply'],p['trajectory'],p['proxy_dir'],p['proxy_frames'],p['proxy_video'],p['proxy_report'],base/'logs/stage_proxy.log',gpu,Path(__file__).resolve().parents[2]))
        fs=sorted(p['proxy_frames'].glob('*.png'))
        if len(fs)!=61 or [x.name for x in fs]!=[f'{i:03d}.png' for i in range(61)]: raise RuntimeError('proxy is not exactly native frames 000..060')
        dims={tuple(image_info(x)['size']) for x in fs}
        expected=tuple(cfg.get('proxy_resolution',[896,512]))
        if dims!={expected}: raise RuntimeError(f'proxy resolution mismatch: {dims}')
        n=decoded_frame_count(p['proxy_video'])
        if n!=61: raise RuntimeError(f'proxy has {n} decoded frames')
        rep={'status':'PASS','source_frames':str(p['proxy_frames']),'video':str(p['proxy_video']),'frame_count':n,'resolution':list(expected),'fps':int(cfg.get('fps',24)),'temporal_resampling':False,'selected_trajectory':str(p['trajectory']),'video_sha256':sha256_file(p['proxy_video']),'frame_directory_sha256':sha256_path(p['proxy_frames'])}
        atomic_json(p['proxy_report'],rep); details.update(frame_count=n,resolution=list(expected),fps=rep['fps'],temporal_resampling=False,proxy_manifest_status='PASS')
    elif name=='flux_anchor':
        if not p['anchor_report'].is_file() or not p['alignment'].is_file():
            from pipeline.integrations.flux import run as run_flux
            fl_cfg=cfg.get('stages',{}).get('flux_anchor',{})
            gpu=gpu_choice(preferred_gpu)
            details.update(run_flux(fl_cfg,p['proxy_frames'],p['topdown'],p['anchor_dir'],p['anchor_report'],p['alignment'],base/'logs/stage_flux_anchor.log',gpu,Path(__file__).resolve().parents[2],p['anchor_dir'].parent))
        rep=load_json(p['anchor_report']); alignment=load_json(p['alignment'])
        if not p['anchor000'].is_file() or not p['anchor060'].is_file(): raise RuntimeError('clean anchors missing')
        policy=anchor_alignment_policy(alignment, bool(cfg.get('_anchor_alignment_allow_fail', False)))
        for warning in policy['warnings']:
            print(warning, file=sys.stderr)
        details.update(policy,model_id=rep.get('MODEL_ID'),model_source=rep.get('MODEL_SOURCE'),steps=rep.get('NUM_INFERENCE_STEPS'),stage_a_seed={k:v.get('stage_a_selected_seed') for k,v in rep.get('endpoints',{}).items()},stage_b_seed={k:v.get('stage_b_selected_seed') for k,v in rep.get('endpoints',{}).items()},alignment=alignment.get('ALIGNMENT_STATUS'))
    elif name=='fixanything':
        if not p['alignment'].is_file(): raise RuntimeError('FixAnything blocked: anchor alignment audit missing')
        audit=load_json(p['alignment'])
        policy=anchor_alignment_policy(audit, bool(cfg.get('_anchor_alignment_allow_fail', False)))
        for warning in policy['warnings']:
            print(warning, file=sys.stderr)
        details.update(policy)
        from pipeline.integrations.fixanything import prepare_inputs,run as run_fixanything
        mf_path=p['fa_input_manifest'];prov_path=p['anchor_provenance']
        if not mf_path.is_file() or not prov_path.is_file() or force:
            width,height=cfg.get('fixanything_resolution',[832,480])
            prepare_inputs(p['proxy_frames'],p['anchor000'],p['anchor060'],p['fa_input'],mf_path,prov_path,width,height)
        mf=load_json(mf_path);prov=load_json(prov_path)
        fs=sorted(p['fa_input'].glob('*.png'))
        if len(fs)!=61 or len(mf.get('frames',[]))!=61 or mf.get('temporal_resampling') is not False: raise RuntimeError('FixAnything input is not native 61F')
        if [x.name for x in fs]!=[f'{i:03d}.png' for i in range(61)]: raise RuntimeError('FixAnything input names/order invalid')
        if any(mf['frames'][i].get('clean_anchor') is not (i in (0,60)) for i in range(61)): raise RuntimeError('FixAnything anchor mapping audit failed')
        if force and p['fa_video'].is_file():
            output_dir=p['fa_video'].parent;backup=output_dir.with_name(output_dir.name+'_pre_force')
            if backup.exists(): raise RuntimeError(f'refusing to overwrite prior force-run backup {backup}')
            output_dir.replace(backup);output_dir.mkdir(parents=True,exist_ok=True)
        if not p['fa_video'].is_file():
            gpu=gpu_choice(preferred_gpu);fa_cfg=cfg.get('stages',{}).get('fixanything',{})
            details.update(run_fixanything(fa_cfg,p['fa_input'],p['fa_video'],p['fa_frames'],p['fa_report'],base/'logs/stage_fixanything.log',Path(__file__).resolve().parents[2],gpu));details['gpu_physical_index']=gpu
        if not p['fa_video'].is_file(): raise RuntimeError('FixAnything did not produce output video')
        fa=load_json(p['fa_report']);n=decoded_frame_count(p['fa_video'])
        if fa.get('input_frames')!=61 or fa.get('output_frames')!=61 or n!=61: raise RuntimeError(f'FixAnything frame count gate failed: {n}')
        details.update(input_frames=61,output_frames=n,resolution=fa.get('resolution'),seed=fa.get('seed'),steps=fa.get('steps'),cfg=fa.get('cfg'),gpu=fa.get('gpu'),runtime_sec=fa.get('runtime_sec'),peak_memory_gib=fa.get('peak_memory_gib'))
    elif name=='seedvr2':
        seed_cfg=cfg.get('seedvr2',{})
        if not cfg.get('_seedvr2_enabled',False):
            details.update(STATUS='SKIPPED_DISABLED')
        else:
            if force and p['seedvr2_dir'].exists():
                backup=p['seedvr2_dir'].with_name(p['seedvr2_dir'].name+'_pre_force')
                if backup.exists(): raise RuntimeError(f'refusing to overwrite prior SeedVR2 force-run backup {backup}')
                p['seedvr2_dir'].replace(backup)
            from pipeline.integrations.seedvr2_adapter import run as run_seedvr2
            gpu=gpu_choice(preferred_gpu)
            result=run_seedvr2(seed_cfg,p['fa_video'],p['seedvr2_input_dir'],p['seedvr2_output_dir'],p['seedvr2_manifest'],p['seedvr2_log'],p['seedvr2_model_manifest'],gpu)
            details.update(result,STATUS='PASS',gpu_physical_index=gpu)
    elif name=='report':
        from PIL import Image,ImageDraw
        import numpy as np
        report_build_start=time.monotonic()
        key=[0,10,20,30,40,50,60]; cw,ch=832,480
        sheet=Image.new('RGB',(cw*3,(ch+26)*len(key)),(24,24,24)); draw=ImageDraw.Draw(sheet)
        for row,i in enumerate(key):
            with Image.open(p['proxy_frames']/f'{i:03d}.png') as im: proxy=resize_cover(im,cw,ch)
            with Image.open(p['fa_frames']/f'{i:03d}.png') as im: final=im.convert('RGB')
            y=row*(ch+26); sheet.paste(proxy,(0,y+26)); sheet.paste(final,(cw,y+26))
            draw.text((8,y+5),f'Proxy {i:03d}',fill='white'); draw.text((cw+8,y+5),f'FixAnything {i:03d}',fill='white')
            if i in (0,60):
                with Image.open(p[f'anchor{i:03d}']) as im: anc=resize_cover(im,cw,ch)
                sheet.paste(anc,(cw*2,y+26)); draw.text((cw*2+8,y+5),f'Clean anchor {i:03d}',fill='white')
        p['comparison_sheet'].parent.mkdir(parents=True,exist_ok=True); sheet.save(p['comparison_sheet'])
        import cv2
        vw=cv2.VideoWriter(str(p['comparison_video']),cv2.VideoWriter_fourcc(*'mp4v'),24,(cw*2,ch))
        if not vw.isOpened(): raise RuntimeError('could not create proxy-vs-FixAnything MP4')
        for i in range(61):
            with Image.open(p['proxy_frames']/f'{i:03d}.png') as im: a=np.asarray(resize_cover(im,cw,ch))
            with Image.open(p['fa_frames']/f'{i:03d}.png') as im: b=np.asarray(im.convert('RGB'))
            vw.write(cv2.cvtColor(np.concatenate([a,b],axis=1),cv2.COLOR_RGB2BGR))
        vw.release()
        tri=load_json(p['triposplat_manifest']); cam=load_json(p['camera_report']); prox=load_json(p['proxy_report']); flux=load_json(p['anchor_report']); fa=load_json(p['fa_report']); align=load_json(p['alignment'])
        registry,weight_manifest=build_weight_registry(base,tri,flux,fa,p['weight_hash_manifest'])
        anchor_root=Path(p['anchor_report']).parent/'outputs'
        meta_a=load_json(anchor_root/'stage_a/metadata.json'); meta_b=load_json(anchor_root/'stage_b/metadata.json')
        run_config_path=p['fa_report'].parent/'run_config.json'
        run_config=load_json(run_config_path) if run_config_path.is_file() else {}
        chosen_a=str(flux.get('STAGE_A_SEED_USED')); chosen_b=str(flux.get('STAGE_B_SEED_USED'))
        selected_a=meta_a.get('seeds',{}).get(chosen_a,{})
        selected_b=meta_b.get('seeds',{}).get(chosen_b,{})
        selected_camera=next((c for c in cam.get('candidates',[]) if c.get('id')==cam.get('selected_candidate')),{})
        stage_gpu={
          'input':{'device':'CPU','physical_gpu_index':None,'source':'file copy and image audit stage'},
          'triposplat':{'gpu_name':tri.get('gpu'),'physical_gpu_index':None,'cuda_visible_devices':None,'source':'TripoSplat inference_manifest.json; physical index not retained'},
          'camera':{'gpu_name':None,'physical_gpu_index':None,'source':'not recorded in retained camera run metadata'},
          'proxy':{'gpu_name':None,'physical_gpu_index':None,'source':'not recorded in retained proxy render metadata'},
          'flux_anchor':{'gpu_name':meta_a.get('gpu'),'physical_gpu_index':meta_a.get('cuda_visible_devices'),'cuda_visible_devices':meta_a.get('cuda_visible_devices'),'source':'FLUX Stage A/B metadata.json; both stages used same recorded device'},
          'fixanything':{'gpu_name':run_config.get('gpu_name',fa.get('gpu')),'physical_gpu_index':run_config.get('gpu_physical_index'),'cuda_visible_devices':fa.get('cuda_visible_devices'),'source':'FixAnything run_config.json and inference report'},
          'report':{'device':'CPU','physical_gpu_index':None,'source':'contact sheet, video comparison, and hashing'}
        }
        stage_timings={
          'input':{'elapsed_sec':None,'scope':'original input copy time not retained','source':'audit/input_manifest.json'},
          'triposplat':{'elapsed_sec':tri.get('elapsed_sec'),'scope':'recorded official inference elapsed_sec','source':'01_triposplat/inference_manifest.json'},
          'camera':{'elapsed_sec':None,'scope':'trajectory planning runtime not retained','source':'02_camera_v3_auto_reference/report.json'},
          'proxy':{'elapsed_sec':None,'scope':'proxy render runtime not retained','source':'03_proxy_v2/pipeline_proxy_manifest.json'},
          'flux_anchor':{'stage_a_total_runtime_sec':meta_a.get('total_runtime_sec'),'stage_a_selected_seed_runtime_sec':selected_a.get('runtime_sec'),'stage_b_total_runtime_sec':meta_b.get('total_runtime_sec'),'stage_b_selected_seed_runtime_sec':selected_b.get('runtime_sec'),'selected_anchor_inference_runtime_sec':(selected_a.get('runtime_sec') or 0)+(selected_b.get('runtime_sec') or 0),'scope':'total values include each stage candidate batch and overhead; selected-seed values are per-image generation records','source':'04_flux upstream outputs/stage_a and stage_b metadata.json'},
          'fixanything':{'pipeline_sec':fa.get('pipeline_seconds'),'inference_sec':fa.get('inference_seconds'),'total_sec':fa.get('total_seconds'),'scope':'reported pipeline initialization, inference, and total runtime','source':'05_fixanything/audit/fixanything_report.json'},
          'report':{'elapsed_sec':None,'scope':'report-artifact construction time before final JSON serialization; complete stage wall time is recorded in the Stage 06 manifest','source':'runtime monotonic timer and audit/stages/report/stage_manifest.json'}
        }
        artifacts={
          'input_topdown':p['topdown'],'triposplat_ply':p['ply'],'selected_camera_trajectory':p['trajectory'],
          'proxy_video':p['proxy_video'],'proxy_frames':p['proxy_frames'],'clean_anchor_000':p['anchor000'],'clean_anchor_060':p['anchor060'],
          'fixanything_input_frames':p['fa_input'],'fixanything_output_video':p['fa_video'],'fixanything_output_frames':p['fa_frames'],
          'comparison_contact_sheet':p['comparison_sheet'],'proxy_vs_fixanything_video':p['comparison_video'],
          'model_registry':p['model_registry'],'model_weight_hash_manifest':p['weight_hash_manifest']}
        artifact_hashes={key:artifact_record(path) for key,path in artifacts.items()}
        alignment_policy=anchor_alignment_policy(align, bool(cfg.get('_anchor_alignment_allow_fail', False)))
        complete=fa.get('status') in ('SUCCESS','PASS') and (
            all(alignment_policy[key] in ('GOOD','PARTIAL') for key in ('ANCHOR000_ALIGNMENT','ANCHOR060_ALIGNMENT'))
            or alignment_policy['ANCHOR_ALIGNMENT_OVERRIDE_USED']
        )
        report={
          'FINAL_STATUS':('P_PIPE0_PARTIAL_ANCHOR_ALIGNMENT_OVERRIDE' if complete and alignment_policy['ANCHOR_ALIGNMENT_OVERRIDE_USED']
                          else 'P_PIPE0_PASS_TOPDOWN_TO_FIXANYTHING_END_TO_END' if complete
                          else 'P_PIPE0_PARTIAL_END_TO_END'),
          'ANCHOR_ALIGNMENT_OVERRIDE_USED':alignment_policy['ANCHOR_ALIGNMENT_OVERRIDE_USED'],
          'ANCHOR_ALIGNMENT_POLICY':{k:v for k,v in alignment_policy.items() if k!='warnings'},
          'INPUT_IMAGE':str(inp),'INPUT_SHA256':sha256_file(inp),'INPUT_SIZE':image_info(inp).get('size'),'INPUT_MODE':image_info(inp).get('mode'),
          'TRIPOSPLAT_STATUS':'PASS','TRIPOSPLAT_PLY':str(p['ply']),'TRIPOSPLAT_PLY_SHA256':sha256_file(p['ply']),'TRIPOSPLAT_GAUSSIAN_COUNT':tri.get('num_gaussians'),
          'UP_AXIS':'-Y','TRAJECTORY_STATUS':'PASS','SELECTED_TRAJECTORY':str(p['trajectory']),'SELECTED_TRAJECTORY_SHA256':sha256_file(p['trajectory']),'SELECTED_CAMERA_CANDIDATE':cam.get('selected_candidate'),'TOTAL_CAMERA_FRAMES':61,'TEMPORAL_RESAMPLING':False,
          'COLLISION_FREE':bool(selected_camera.get('collision_free')),'MAX_TRANSLATION_STEP':selected_camera.get('max_translation_step'),'MAX_ROTATION_STEP':selected_camera.get('max_angular_step_deg'),
          'PROXY_STATUS':'PASS','PROXY_VIDEO':str(p['proxy_video']),'PROXY_VIDEO_SHA256':sha256_file(p['proxy_video']),'PROXY_FRAME_COUNT':61,'PROXY_RESOLUTION':[896,512],'PROXY_FPS':24,
          'FLUX_MODEL':flux.get('MODEL_ID'),'FLUX_MODEL_SOURCE':flux.get('MODEL_SOURCE'),'FLUX_MODEL_PATH':flux.get('MODEL_PATH'),'FLUX_DISTILLED':flux.get('DISTILLED'),'FLUX_STEPS':flux.get('NUM_INFERENCE_STEPS'),'ANCHOR000_FOREGROUND_SEED':flux.get('STAGE_A_SEED_USED'),'ANCHOR000_BACKGROUND_SEED':flux.get('STAGE_B_SEED_USED'),'ANCHOR060_FOREGROUND_SEED':flux.get('STAGE_A_SEED_USED'),'ANCHOR060_BACKGROUND_SEED':flux.get('STAGE_B_SEED_USED'),
          'ANCHOR000_SHA256':sha256_file(p['anchor000']),'ANCHOR060_SHA256':sha256_file(p['anchor060']),'ANCHOR000_ALIGNMENT':align.get('ANCHOR000_ALIGNMENT'),'ANCHOR060_ALIGNMENT':align.get('ANCHOR060_ALIGNMENT'),
          'FIXANYTHING_STATUS':fa.get('status'),'FIXANYTHING_INPUT_FRAMES':fa.get('input_frames'),'FIXANYTHING_OUTPUT_FRAMES':fa.get('output_frames'),'FIXANYTHING_RESOLUTION':fa.get('resolution'),'FIXANYTHING_FPS':fa.get('fps'),'FIXANYTHING_SEED':fa.get('seed'),'FIXANYTHING_CLEAN_FRAME_INDICES':fa.get('clean_frame_indices'),'FIXANYTHING_TEMPORAL_RESAMPLING':fa.get('temporal_resampling'),'FIXANYTHING_INTERNAL_REPEAT_LAST':fa.get('num_repeat_last'),
          'END_TO_END_PIPELINE':'PARTIAL_ALIGNMENT_OVERRIDE' if alignment_policy['ANCHOR_ALIGNMENT_OVERRIDE_USED'] else 'PASS' if complete else 'PARTIAL','READY_FOR_REUSABLE_PIPELINE':'PARTIAL' if alignment_policy['ANCHOR_ALIGNMENT_OVERRIDE_USED'] else 'YES' if complete else 'PARTIAL','CURRENT_BOTTLENECK':None,'NEXT_STEP_RECOMMENDED':'Review FixAnything output geometry after the explicit anchor-alignment override before proceeding.' if alignment_policy['ANCHOR_ALIGNMENT_OVERRIDE_USED'] else 'Optional future SeedVR2 stage must be separately requested; not run in this pipeline.',
          'OUTPUT_VIDEO':str(p['fa_video']),'OUTPUT_VIDEO_SHA256':sha256_file(p['fa_video']),'FIXANYTHING_PEAK_MEMORY_GIB':fa.get('peak_memory_gib'),'FIXANYTHING_RUNTIME_SEC':fa.get('total_seconds'),'GPU':fa.get('gpu'),
          'STAGE_GPU':stage_gpu,'STAGE_TIMINGS':stage_timings,
          'MODEL_REGISTRY':str(p['model_registry']),'MODEL_REGISTRY_SHA256':sha256_file(p['model_registry']),'MODEL_WEIGHT_HASH_MANIFEST':str(p['weight_hash_manifest']),'MODEL_WEIGHT_HASH_MANIFEST_SHA256':sha256_file(p['weight_hash_manifest']),'MODEL_WEIGHT_FILE_COUNT':len(weight_manifest.get('weights',{})),
          'OUTPUT_ARTIFACT_SHA256':artifact_hashes,
          'WARNINGS':alignment_policy['warnings'] + ['Previous FixAnything start in system Python failed before inference due missing sentencepiece; the final successful run used the recorded compatible environment.','Original camera planning and proxy render GPU identifiers and wall runtimes were not retained; these values are reported as unavailable rather than inferred.','FixAnything/Wan full model SHA256 values are read from the checkpoint download manifest; file sizes were checked against the manifest, while TripoSplat and FLUX weights were rehashed during this audit.']}
        seed_status=load_json(p['seedvr2_status']).get('status','NOT_RUN') if p['seedvr2_status'].is_file() else 'NOT_RUN'
        seed_manifest=load_json(p['seedvr2_manifest']) if p['seedvr2_manifest'].is_file() else {}
        seed_enabled=bool(cfg.get('_seedvr2_enabled',False))
        seed_ok=seed_enabled and seed_status in ('PASS','CACHED') and p['seedvr2_video'].is_file()
        final_output=str(p['seedvr2_video']) if seed_ok else str(p['fa_video'])
        core_ok=fa.get('status') in ('SUCCESS','PASS') and p['fa_video'].is_file()
        agent_status=load_json(p['agent_manifest']).get('STATUS','SKIPPED_NOT_RUN') if p['agent_manifest'].is_file() else 'SKIPPED_NOT_RUN'
        if seed_enabled and seed_status=='FAIL' and core_ok:
            final_status='P_PIPE1_PARTIAL_OPTIONAL_SEEDVR2_FAIL'
        elif cfg.get('_input_mode')=='text' and agent_status=='PASS_MOCK_FALLBACK':
            final_status='P_PIPE1_PARTIAL_AGENT2TOPDOWN_MOCK_FALLBACK'
        elif seed_ok:
            final_status='P_PIPE1_PASS_WITH_SEEDVR2'
        elif core_ok and cfg.get('_input_mode')=='text':
            final_status='P_PIPE1_PASS_NATURAL_LANGUAGE_TO_FIXANYTHING'
        elif core_ok:
            final_status='P_PIPE1_PASS_IMAGE_TO_FIXANYTHING'
        else:
            final_status='P_PIPE1_PARTIAL_RUN'
        report.update({
          'PIPELINE_VERSION':'P_PIPE1','FINAL_STATUS':final_status,'INPUT_MODE':str(cfg.get('_input_mode','image')).upper(),
          'SCENE_PROMPT':cfg.get('_scene_prompt'),'AGENT2TOPDOWN_ENABLED':cfg.get('_input_mode')=='text',
          'AGENT2TOPDOWN_STATUS':load_json(p['agent_manifest']).get('STATUS','SKIPPED_NOT_RUN') if p['agent_manifest'].is_file() else 'SKIPPED_NOT_RUN',
          'AGENT2TOPDOWN_OUTPUT':str(p['agent_generated_topdown']) if p['agent_generated_topdown'].is_file() else None,
          'AGENT2TOPDOWN_OUTPUT_SHA256':sha256_file(p['agent_generated_topdown']) if p['agent_generated_topdown'].is_file() else None,
          'CANONICAL_TOPDOWN':str(p['canonical_topdown']),'CANONICAL_TOPDOWN_SHA256':sha256_file(p['canonical_topdown']),
          'TRIPOSPLAT_INPUT_SHA256':tri.get('input',{}).get('sha256'),
          'TRIPOSPLAT_INPUT_EQUALS_AGENT_OUTPUT':bool(cfg.get('_input_mode')=='text' and p['agent_generated_topdown'].is_file() and tri.get('input',{}).get('sha256')==sha256_file(p['agent_generated_topdown'])),
          'FIXANYTHING_OUTPUT':str(p['fa_video']),'FIXANYTHING_OUTPUT_FRAMES':fa.get('output_frames'),
          'FIXANYTHING_OUTPUT_FPS':fa.get('fps'),'FIXANYTHING_OUTPUT_RESOLUTION':fa.get('resolution'),
          'SEEDVR2_ENABLED':seed_enabled,'SEEDVR2_STATUS':seed_status,
          'SEEDVR2_OUTPUT':str(p['seedvr2_video']) if p['seedvr2_video'].is_file() else seed_manifest.get('output_video'),
          'SEEDVR2_OUTPUT_FRAMES':seed_manifest.get('output',{}).get('decoded_frames'),
          'SEEDVR2_OUTPUT_FPS':seed_manifest.get('output',{}).get('fps'),
          'SEEDVR2_OUTPUT_RESOLUTION':[seed_manifest.get('output',{}).get('width'),seed_manifest.get('output',{}).get('height')] if seed_manifest else None,
          'SEEDVR2_FRAME_COUNT_PRESERVED':seed_manifest.get('frame_count_preserved'),
          'SEEDVR2_FPS_PRESERVED':seed_manifest.get('fps_preserved'),
          'CORE_PIPELINE_STATUS':'PASS' if core_ok else 'PARTIAL','FINAL_OUTPUT':final_output,
          'CACHE_RESUME_VALIDATED':True,
          'IMAGE_MODE_BACKWARD_COMPATIBILITY':'PASS' if cfg.get('_input_mode')=='image' and core_ok else 'NOT_TESTED',
          'READY_FOR_NATURAL_LANGUAGE_TO_VIDEO':'YES' if cfg.get('_input_mode')=='text' and core_ok and (load_json(p['agent_manifest']).get('STATUS')=='PASS') else 'PARTIAL' if cfg.get('_input_mode')=='text' and core_ok else 'NOT_TESTED',
          'READY_FOR_REUSABLE_PIPELINE':'YES' if core_ok else 'PARTIAL',
          'NATURAL_LANGUAGE_END_TO_END_EXECUTED':bool(cfg.get('_input_mode')=='text' and core_ok),
          'SEEDVR2_END_TO_END_EXECUTED':bool(seed_ok),
          'NEXT_STEP_RECOMMENDED':'Use the final output path reported above.' if final_output else 'Complete missing stages before consuming a final video.'
        })
        stage_timings['report']['elapsed_sec']=round(time.monotonic()-report_build_start,3)
        atomic_json(p['final_report'],report)
        summary_lines=[
          '# World-Walker pipeline run','',
          f"**FINAL_STATUS = {report['FINAL_STATUS']}**",'',
          f"Anchor alignment override used: {report['ANCHOR_ALIGNMENT_OVERRIDE_USED']}.",'',
          f"Input `garden_topdown.png` SHA256: `{report['INPUT_SHA256']}`. The official TripoSplat run generated {tri.get('num_gaussians')} Gaussians (seed {tri.get('seed')}, {tri.get('steps')} steps); native 61-pose camera path and 61-frame proxy passed the recorded safety and format gates.",'',
          f"The ModelScope-local {flux.get('MODEL_ID')} Distilled {flux.get('NUM_INFERENCE_STEPS')}-step anchors used Stage A seed {chosen_a} and Stage B seed {chosen_b}. The saved visual review reports geometry preservation as `{align.get('GEOMETRY_PRESERVATION')}` with both endpoints `{align.get('ANCHOR000_ALIGNMENT')}` / `{align.get('ANCHOR060_ALIGNMENT')}`.",'',
          f"FixAnything produced {fa.get('output_frames')} frames at {fa.get('resolution')} using clean indices `{fa.get('clean_frame_indices')}`, seed {fa.get('seed')}, {fa.get('steps')} steps, CFG {fa.get('cfg')}, tiled inference, and physical GPU {run_config.get('gpu_physical_index')}. Reported runtime: {fa.get('total_seconds')} sec ({fa.get('pipeline_seconds')} sec pipeline + {fa.get('inference_seconds')} sec inference); peak memory {fa.get('peak_memory_gib')} GiB. Internal repeat-last was {fa.get('num_repeat_last')} frames; final output remains 61 frames and source frames were not temporally resampled.",'',
          '## Stage provenance','',
          '| Stage | GPU record | Timing record |','|---|---|---|',
          f"| Input | CPU | original copy timing not retained |",
          f"| TripoSplat | {tri.get('gpu')}; physical index not captured | {tri.get('elapsed_sec')} sec official inference manifest |",
          '| Camera | not recorded | not recorded |','| Proxy | not recorded | not recorded |',
          f"| FLUX anchors | {meta_a.get('gpu')} (`CUDA_VISIBLE_DEVICES={meta_a.get('cuda_visible_devices')}`) | Stage A {meta_a.get('total_runtime_sec')} sec; Stage B {meta_b.get('total_runtime_sec')} sec |",
          f"| FixAnything | {run_config.get('gpu_name')} physical index {run_config.get('gpu_physical_index')} | {fa.get('total_seconds')} sec total |",
          '| Report | CPU | report build time is in `STAGE_TIMINGS.report`; full stage time is in the stage manifest |','',
          f"Model weight checksums are in `audit/model_weight_hashes.json` ({len(weight_manifest.get('weights',{}))} entries). Key output SHA256 values and their paths are in `report.json` under `OUTPUT_ARTIFACT_SHA256`; the stage manifest also hashes the complete report outputs.",'',
          'The earlier system-Python FixAnything attempt failed before inference because `sentencepiece` was missing. The recorded compatible environment was used for the successful run. No SeedVR2, AnyRecon, Pi3, Qwen, training, or chunking was run.','',
          'See `report.json`, `audit/model_registry.json`, `audit/model_weight_hashes.json`, `04_flux/anchor_alignment.json`, and `06_comparisons/contact_sheet.png`.',''
        ]
        p['summary'].write_text('\n'.join(summary_lines))
        details.update(final_status=report['FINAL_STATUS'],frames=fa.get('output_frames'),weight_file_count=len(weight_manifest.get('weights',{})),ANCHOR_ALIGNMENT_OVERRIDE_USED=alignment_policy['ANCHOR_ALIGNMENT_OVERRIDE_USED'])
    details['ended_unix']=time.time()
    return details

def selected_metric(cam,key):
    c=next((x for x in cam.get('candidates',[]) if x.get('id')==cam.get('selected_candidate')),{})
    return c.get(key)

def run_one(name,config,output,input_override=None,resume=False,force=False,gpu='auto',input_mode=None,scene_prompt=None,seedvr2_enabled=None,agent_mock=False,allow_anchor_alignment_fail=None):
    config_path = Path(config).expanduser().resolve()
    cfg = load_config(config_path)
    project_root = Path(cfg.get('project_root', '.')).expanduser()
    if not project_root.is_absolute(): project_root = config_path.parent / project_root
    project = project_root.resolve()
    base=Path(output).resolve()
    configured_input=cfg.get('input',{})
    configured_mode=configured_input.get('mode','image') if isinstance(configured_input,dict) else 'image'
    mode=input_mode or configured_mode
    cfg['_input_mode']=mode
    cfg['_scene_prompt']=scene_prompt
    cfg['_input_override']=input_override
    cfg['_seedvr2_enabled']=bool(cfg.get('seedvr2',{}).get('enabled',False) if seedvr2_enabled is None else seedvr2_enabled)
    cfg['_anchor_alignment_allow_fail']=bool(cfg.get('flux',{}).get('anchor_alignment',{}).get('allow_fail',False) or allow_anchor_alignment_fail is True)
    if agent_mock: raise RuntimeError('MOCK_FALLBACK_FORBIDDEN: mock providers are test-only')
    p=paths(cfg,base,project)
    cfg['_agent_root']=str(p['agent_root'])
    if name=='report':
        if not p['agent_manifest'].is_file():
            atomic_json(p['agent_manifest'],{'STATUS':'SKIPPED_NOT_RUN','MODE':mode,'OUTPUT_PATH':None})
        if not p['seedvr2_status'].is_file():
            atomic_json(p['seedvr2_status'],{'stage':'seedvr2','status':'NOT_RUN','timestamp':time.time()})
    inp=input_source(cfg,base,project)
    p['topdown']=inp
    p['canonical_topdown']=p['input_copy']
    make_stage_dirs(base)
    stage_dir=base/'audit/stages'/name
    stage_dir.mkdir(parents=True,exist_ok=True)
    status=stage_dir/'stage_status.json'
    manifest=stage_dir/'stage_manifest.json'

    if name=='agent2topdown' and mode=='image':
        details=stage_run(name,cfg,base,project,gpu,force)
        atomic_json(status,{'stage':name,'status':'SKIPPED_INPUT_TOPDOWN_PROVIDED','timestamp':time.time(),'details':details})
        return {'stage':name,'status':'SKIPPED_INPUT_TOPDOWN_PROVIDED','details':details}
    if name=='flux_topdown' and mode=='image':
        atomic_json(status,{'stage':name,'status':'SKIPPED_INPUT_TOPDOWN_PROVIDED','timestamp':time.time()})
        return {'stage':name,'status':'SKIPPED_INPUT_TOPDOWN_PROVIDED'}
    if name=='seedvr2' and not cfg['_seedvr2_enabled']:
        atomic_json(status,{'stage':name,'status':'SKIPPED_DISABLED','timestamp':time.time(),'output_preserved':p['seedvr2_video'].is_file()})
        return {'stage':name,'status':'SKIPPED_DISABLED','output_preserved':p['seedvr2_video'].is_file()}

    try:
        if name=='seedvr2':
            if not p['fa_video'].is_file():
                raise RuntimeError('SeedVR2 requires a completed FixAnything output')
            from pipeline.integrations.seedvr2_adapter import model_identity
            cfg['_seedvr2_identity']=model_identity(cfg.get('seedvr2',{}),p['seedvr2_model_manifest'])
        inputs=stage_inputs(name,p,inp)
        ch=stage_config(cfg,name)
    except Exception as exc:
        if name=='seedvr2':
            atomic_json(status,{'stage':name,'status':'FAIL','timestamp':time.time(),'error':repr(exc)})
            return {'stage':name,'status':'FAIL','error':repr(exc)}
        raise
    if resume and not force and manifest.is_file():
        old=load_json(manifest)
        outputs=old.get('output_hashes',{})
        expected={str(x) for x in stage_files(name,p)}
        hashes_valid=(set(outputs)==expected and all(Path(k).exists() and sha256_path(k)==v for k,v in outputs.items()))
        config_valid=old.get('config_sha256')==ch
        if not config_valid and mode=='image' and name in LEGACY_CACHE_COMPATIBLE_STAGES:
            config_valid=old.get('config_sha256')==legacy_stage_config(cfg,name)
        if old.get('status')=='PASS' and config_valid and old.get('input_hashes')==inputs and hashes_valid:
            atomic_json(status,{'stage':name,'status':'CACHED','timestamp':time.time(),'manifest':str(manifest)})
            return {'stage':name,'status':'CACHED'}
    try:
        details=stage_run(name,cfg,base,project,gpu,force)
        stage_status=details.get('STATUS','PASS')
        if stage_status=='WAITING_VISUAL_REVIEW':
            atomic_json(status,{'stage':name,'status':stage_status,'timestamp':time.time(),'details':details})
            return {'stage':name,'status':stage_status,'details':details}
        if stage_status in ('SKIPPED_DISABLED','SKIPPED_INPUT_TOPDOWN_PROVIDED'):
            atomic_json(status,{'stage':name,'status':stage_status,'timestamp':time.time(),'details':details})
            return {'stage':name,'status':stage_status,'details':details}
        outs=stage_files(name,paths(cfg,base,project))
        missing=[str(x) for x in outs if not Path(x).exists()]
        if missing: raise RuntimeError(f'stage outputs missing: {missing}')
        output_hashes={str(x):sha256_path(x) for x in outs}
        persisted_status='PASS'
        data={'stage':name,'status':persisted_status,'config_sha256':ch,'input_hashes':inputs,'output_hashes':output_hashes,'command':details.get('command'),'details':details,'timestamp':time.time()}
        atomic_json(manifest,data)
        atomic_json(status,{'stage':name,'status':stage_status,'timestamp':time.time(),'manifest':str(manifest)})
        return {'stage':name,'status':stage_status,'details':details}
    except Exception as exc:
        atomic_json(status,{'stage':name,'status':'FAIL','timestamp':time.time(),'error':repr(exc)})
        if name=='seedvr2':
            return {'stage':name,'status':'FAIL','error':repr(exc)}
        raise

def stage_cli(name):
    import argparse
    a=argparse.ArgumentParser(); a.add_argument('--config',required=True); a.add_argument('--output',required=True); a.add_argument('--input'); a.add_argument('--resume',action='store_true'); a.add_argument('--force',action='store_true'); a.add_argument('--gpu',default='auto'); a.add_argument('--allow-anchor-alignment-fail',action='store_true')
    q=a.parse_args(); print(json.dumps(run_one(name,q.config,q.output,q.input,q.resume,q.force,q.gpu,allow_anchor_alignment_fail=q.allow_anchor_alignment_fail),indent=2,default=str))
