#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, sys
HERE=Path(__file__).resolve().parent
from pipeline.integrations.agent2topdown_adapter import resolve_input_mode
from pipeline.utils.runtime import EXECUTION_ORDER,LEGACY_NUMERIC_STAGES,atomic_json,load_json,paths,run_one
from pipeline.utils.paths import load_config

def stage_id(value):
    stage=str(value).lower().replace("stage","").strip()
    numeric=stage.lstrip("0") or "0"
    if numeric=="7": return "seedvr2"
    if numeric=="8": return "report"
    if numeric in LEGACY_NUMERIC_STAGES: return LEGACY_NUMERIC_STAGES[numeric]
    stage={"agent":"agent2topdown","topdown":"flux_topdown","flux":"flux_anchor","fix":"fixanything"}.get(stage,stage)
    if stage in EXECUTION_ORDER: return stage
    raise argparse.ArgumentTypeError(f"unknown stage: {value}; choose {', '.join(EXECUTION_ORDER)} or legacy 00-06")

def resolve_project_root(config_path, cfg):
    root = Path(cfg.get("project_root", ".")).expanduser()
    if not root.is_absolute(): root = Path(config_path).expanduser().resolve().parent / root
    return root.resolve()

def should_finalize_report(results):
    return any(item.get("stage")=="report" for item in results)

def build_parser():
    parser=argparse.ArgumentParser(description="World-Walker research pipeline: top-down image or natural-language input.")
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--input-topdown","--input",dest="input_topdown",help="existing topdown image (legacy --input alias)")
    modes.add_argument("--scene-prompt",help="natural-language garden scene description")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/default.yaml"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--start-stage",type=stage_id,default="0")
    parser.add_argument("--stop-stage",type=stage_id,default="6")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--allow-anchor-alignment-fail",action="store_true", help="continue after a failed FLUX anchor alignment audit; use only after visually inspecting the anchor/proxy comparison")
    parser.add_argument("--force-stage",action="append",type=stage_id,default=[])
    parser.add_argument("--gpu",default="auto")
    seed=parser.add_mutually_exclusive_group()
    seed.add_argument("--enable-seedvr2",dest="seedvr2_override",action="store_true")
    seed.add_argument("--disable-seedvr2",dest="seedvr2_override",action="store_false")
    parser.set_defaults(seedvr2_override=None)
    parser.add_argument("--dry-run", action="store_true", help="validate configuration and print planned stages without running models")
    return parser

def select_final_output(seedvr2_enabled,seedvr2_status,seedvr2_video,fixanything_video):
    if seedvr2_enabled and seedvr2_status in ("PASS","CACHED") and Path(seedvr2_video).is_file():
        return str(seedvr2_video)
    return str(fixanything_video) if Path(fixanything_video).is_file() else None

def _finalize_report(config_path,output,mode,prompt,seedvr2_enabled,results,allow_anchor_alignment_fail=False):
    report_path=Path(output).resolve()/"report.json"
    summary_path=Path(output).resolve()/"summary.md"
    if not report_path.is_file(): return None
    report=load_json(report_path)
    cfg=load_config(config_path); project=resolve_project_root(config_path, cfg)
    cfg["_input_mode"]=mode; cfg["_scene_prompt"]=prompt
    p=paths(cfg,Path(output).resolve(),project)
    cfg["_anchor_alignment_allow_fail"]=bool(cfg.get("flux",{}).get("anchor_alignment",{}).get("allow_fail",False) or allow_anchor_alignment_fail)
    agent=load_json(p["agent_manifest"]) if p["agent_manifest"].is_file() else {}
    topdown_refinement=load_json(p["flux_topdown_manifest"]) if p["flux_topdown_manifest"].is_file() else {}
    seed=load_json(p["seedvr2_manifest"]) if p["seedvr2_manifest"].is_file() else {}
    fa=load_json(p["fa_report"]) if p["fa_report"].is_file() else {}
    alignment=load_json(p["alignment"]) if p["alignment"].is_file() else {}
    from pipeline.utils.runtime import anchor_alignment_policy
    alignment_policy=anchor_alignment_policy(alignment,bool(cfg["_anchor_alignment_allow_fail"])) if alignment else {"ANCHOR_ALIGNMENT_OVERRIDE_USED":False,"warnings":[]}
    override_used=bool(alignment_policy["ANCHOR_ALIGNMENT_OVERRIDE_USED"])
    warnings=list(report.get("WARNINGS",[]))
    for warning in alignment_policy.get("warnings",[]):
        if warning not in warnings:
            warnings.append(warning)
    tri=load_json(p["triposplat_manifest"]) if p["triposplat_manifest"].is_file() else {}
    seed_status=load_json(p["seedvr2_status"]).get("status","NOT_RUN") if p["seedvr2_status"].is_file() else "NOT_RUN"
    canonical=p["canonical_topdown"]
    canonical_sha=None
    if canonical.is_file():
        from pipeline.utils.hashing import sha256_file
        canonical_sha=sha256_file(canonical)
    core_pass=bool(fa.get("status") in ("SUCCESS","PASS") and p["fa_video"].is_file())
    agent_status=agent.get("STATUS","SKIPPED_NOT_RUN")
    final_output=select_final_output(seedvr2_enabled,seed_status,p["seedvr2_video"],p["fa_video"])
    if override_used and core_pass:
        final_status="P_PIPE1_PARTIAL_ANCHOR_ALIGNMENT_OVERRIDE"
    elif seedvr2_enabled and seed_status=="FAIL" and core_pass:
        final_status="P_PIPE1_PARTIAL_OPTIONAL_SEEDVR2_FAIL"
    elif not core_pass:
        final_status="P_PIPE1_PARTIAL_RUN"
    elif mode=="text" and agent_status=="PASS_MOCK_FALLBACK":
        final_status="P_PIPE1_PARTIAL_AGENT2TOPDOWN_MOCK_FALLBACK"
    elif mode=="text" and agent_status=="PASS_MOCK_FALLBACK":
        final_status="P_PIPE1_PARTIAL_AGENT2TOPDOWN_MOCK_FALLBACK"
    elif seedvr2_enabled and seed_status in ("PASS","CACHED"):
        final_status="P_PIPE1_PASS_WITH_SEEDVR2"
    elif mode=="text":
        final_status="P_PIPE1_PASS_NATURAL_LANGUAGE_TO_FIXANYTHING"
    else:
        final_status="P_PIPE1_PASS_IMAGE_TO_FIXANYTHING"
    report.update({
        "PROJECT_NAME":"World-Walker","PIPELINE_VERSION":"P_RELEASE2","FINAL_STATUS":final_status,"INPUT_MODE":mode.upper(),"SCENE_PROMPT":prompt,
        "ANCHOR000_ALIGNMENT":alignment.get("ANCHOR000_ALIGNMENT"),"ANCHOR060_ALIGNMENT":alignment.get("ANCHOR060_ALIGNMENT"),
        "ANCHOR_ALIGNMENT_OVERRIDE_USED":override_used,"ANCHOR_ALIGNMENT_OVERRIDE_ENABLED":bool(cfg["_anchor_alignment_allow_fail"]),
        "WARNINGS":warnings,
        "AGENT2TOPDOWN_ENABLED":mode=="text","AGENT2TOPDOWN_STATUS":agent_status,
        "AGENT2TOPDOWN_OUTPUT":agent.get("OUTPUT_PATH"),"AGENT2TOPDOWN_OUTPUT_SHA256":agent.get("OUTPUT_SHA256"),
        "AGENT2TOPDOWN_MODE":agent.get("MODE"),"AGENT2TOPDOWN_MODEL_OR_SERVICE":agent.get("MODEL_OR_SERVICE_INFO"),
        "FLUX_TOPDOWN_STATUS":load_json(p["flux_topdown_status"]).get("status","NOT_RUN") if p["flux_topdown_status"].is_file() else "NOT_RUN",
        "FLUX_TOPDOWN_MODEL_ID":topdown_refinement.get("model_id"),"FLUX_TOPDOWN_MODEL_SOURCE":topdown_refinement.get("model_source"),
        "FLUX_TOPDOWN_SELECTED_SEED":topdown_refinement.get("selected_seed"),"FLUX_TOPDOWN_STEPS":topdown_refinement.get("num_inference_steps"),
        "STRICT_NADIR_VISUAL_REVIEW":topdown_refinement.get("strict_nadir_view"),"TOPDOWN_USED":str(p["topdown_used"]) if p["topdown_used"].is_file() else None,
        "TOPDOWN_USED_SHA256":sha256_file(p["topdown_used"]) if p["topdown_used"].is_file() else None,
        "CANONICAL_TOPDOWN":str(canonical),"CANONICAL_TOPDOWN_SHA256":canonical_sha,
        "TRIPOSPLAT_INPUT_SHA256":tri.get("input",{}).get("sha256"),
        "TRIPOSPLAT_INPUT_EQUALS_AGENT_OUTPUT":bool(mode=="text" and tri.get("input",{}).get("sha256")==agent.get("OUTPUT_SHA256")),
        "TRIPOSPLAT_INPUT_EQUALS_TOPDOWN_USED":bool(mode=="text" and p["topdown_used"].is_file() and tri.get("input",{}).get("sha256")==sha256_file(p["topdown_used"])),
        "FIXANYTHING_OUTPUT":str(p["fa_video"]) if p["fa_video"].is_file() else None,
        "FIXANYTHING_OUTPUT_FRAMES":fa.get("output_frames"),"FIXANYTHING_OUTPUT_FPS":fa.get("fps"),
        "FIXANYTHING_OUTPUT_RESOLUTION":fa.get("resolution"),
        "SEEDVR2_ENABLED":seedvr2_enabled,"SEEDVR2_STATUS":seed_status,
        "SEEDVR2_OUTPUT":str(p["seedvr2_video"]) if p["seedvr2_video"].is_file() else seed.get("output_video"),
        "SEEDVR2_OUTPUT_FRAMES":seed.get("output",{}).get("decoded_frames"),
        "SEEDVR2_OUTPUT_FPS":seed.get("output",{}).get("fps"),
        "SEEDVR2_OUTPUT_RESOLUTION":[seed.get("output",{}).get("width"),seed.get("output",{}).get("height")] if seed else None,
        "SEEDVR2_FRAME_COUNT_PRESERVED":seed.get("frame_count_preserved"),"SEEDVR2_FPS_PRESERVED":seed.get("fps_preserved"),
        "CORE_PIPELINE_STATUS":"PARTIAL_ALIGNMENT_OVERRIDE" if override_used and core_pass else "PASS" if core_pass else "PARTIAL","FINAL_OUTPUT":final_output,"STAGE_RUN_RESULTS":results,
        "NATURAL_LANGUAGE_END_TO_END_EXECUTED":bool(mode=="text" and agent_status=="PASS" and core_pass),
        "SEEDVR2_END_TO_END_EXECUTED":bool(seedvr2_enabled and seed_status in ("PASS","CACHED")),
        "READY_FOR_NATURAL_LANGUAGE_TO_VIDEO":"PARTIAL" if override_used and mode=="text" and core_pass else "YES" if mode=="text" and agent_status=="PASS" and core_pass else "PARTIAL" if mode=="text" and agent_status=="PASS_MOCK_FALLBACK" else "NO" if mode=="text" else "NOT_TESTED",
        "READY_FOR_REUSABLE_PIPELINE":"PARTIAL" if override_used else "YES" if core_pass and (mode!="text" or agent_status=="PASS") else "PARTIAL",
        "NEXT_STEP_RECOMMENDED":"Review FixAnything output geometry after the explicit anchor-alignment override before proceeding." if override_used else "Provide a DeepSeek API key and repeat text-mode integration; current output used the built-in mock." if mode=="text" and agent_status=="PASS_MOCK_FALLBACK" else "Use the final output path reported above."
    })
    atomic_json(report_path,report)
    lines=["# World-Walker pipeline run","",f"FINAL_STATUS = {final_status}","",
           f"- Anchor alignment override used: {override_used} (enabled: {bool(cfg['_anchor_alignment_allow_fail'])})",
           f"- Input mode: {mode.upper()}",
           f"- Agent2Topdown: {agent_status} (mode: {agent.get('MODE')})",
           f"- Canonical topdown SHA256: {canonical_sha}",
           f"- TripoSplat input SHA256: {tri.get('input',{}).get('sha256')}",
           f"- FixAnything: {fa.get('status')}, output {p['fa_video'] if p['fa_video'].is_file() else 'not produced'}",
           f"- SeedVR2: enabled {seedvr2_enabled}, status {seed_status}",f"- Final output: {final_output}","",
           "## Stage results","","| Stage | Status |","|---|---|"]
    lines.extend(f"| {item.get('stage')} | {item.get('status')} |" for item in results)
    lines.extend(["","See report.json, audit/model_registry.json, and audit/stages/.",""])
    summary_path.write_text("\n".join(lines),encoding="utf-8")
    manifest_path=Path(output).resolve()/"audit/stages/report/stage_manifest.json"
    if manifest_path.is_file():
        from pipeline.utils.hashing import sha256_path
        manifest=load_json(manifest_path)
        manifest["output_hashes"]={str(path):sha256_path(path) for path in [
            p["final_report"],p["summary"],p["comparison_sheet"],p["comparison_video"],p["model_registry"],p["weight_hash_manifest"]
        ] if Path(path).exists()}
        atomic_json(manifest_path,manifest)
    return report

def main(argv=None):
    parser=build_parser(); args=parser.parse_args(argv)
    try: mode,selected_input=resolve_input_mode(args.input_topdown,args.scene_prompt)
    except ValueError as exc: parser.error(str(exc))
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    project = resolve_project_root(config_path, cfg)
    input_cfg = cfg.get("input", {})
    configured_mode = input_cfg.get("mode", "auto") if isinstance(input_cfg, dict) else "auto"
    if configured_mode not in ("auto", mode):
        parser.error(f"input mode mismatch: CLI selected {mode!r}, config declares {configured_mode!r}")
    if mode=="image" and not Path(selected_input).expanduser().is_file():
        image_path = Path(selected_input).expanduser()
        if not image_path.is_absolute(): image_path = project / image_path
        if not image_path.is_file() and not args.dry_run: parser.error(f"topdown image does not exist: {selected_input}")
    seed_enabled=cfg.get("seedvr2",{}).get("enabled",False)
    if args.seedvr2_override is not None: seed_enabled=args.seedvr2_override
    anchor_override=bool(cfg.get("flux",{}).get("anchor_alignment",{}).get("allow_fail",False) or args.allow_anchor_alignment_fail)
    start=args.start_stage; stop=args.stop_stage
    positions={name:index for index,name in enumerate(EXECUTION_ORDER)}
    if stop=="agent2topdown" and start=="input": start="agent2topdown"
    if positions[start]>positions[stop]: parser.error("start-stage must not come after stop-stage")
    selected_stages=[name for name in EXECUTION_ORDER if positions[start]<=positions[name]<=positions[stop]]
    if positions[start]<=positions["input"]:
        if "agent2topdown" not in selected_stages: selected_stages.insert(0,"agent2topdown")
        if mode=="text" and "flux_topdown" not in selected_stages:
            selected_stages.insert(selected_stages.index("input"),"flux_topdown")
    if stop=="seedvr2" and "report" not in selected_stages: selected_stages.append("report")
    force = set(args.force_stage); results = []
    if args.dry_run:
        print(json.dumps({
            "DRY_RUN": True, "INPUT_MODE": mode,
            "INPUT": selected_input if mode == "image" else None,
            "SCENE_PROMPT": selected_input if mode == "text" else None,
            "CONFIG": str(config_path), "PROJECT_ROOT": str(project),
            "OUTPUT": str(Path(args.output).expanduser().resolve()),
            "PLANNED_STAGES": selected_stages,
            "NATIVE_FRAMES": int(cfg.get("total_frames", 61)),
            "TEMPORAL_RESAMPLING": bool(cfg.get("temporal_resampling", False)),
            "CLEAN_FRAME_INDICES": cfg.get("stages", {}).get("fixanything", {}).get("clean_frame_indices", [0, 60]),
            "SEEDVR2_ENABLED": bool(seed_enabled), "ANCHOR_ALIGNMENT_FAIL_OVERRIDE_ENABLED":anchor_override, "MODEL_EXECUTION": False
        }, indent=2, ensure_ascii=False))
        return 0
    input_override=selected_input if mode=="image" else None
    prompt=selected_input if mode=="text" else None
    failure = None
    for name in selected_stages:
        try:
            result=run_one(name,args.config,args.output,input_override,args.resume,name in force,args.gpu,
                           input_mode=mode,scene_prompt=prompt,seedvr2_enabled=bool(seed_enabled),allow_anchor_alignment_fail=anchor_override)
        except Exception as exc:
            failure = {"stage": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
            results.append(failure)
            break
        results.append(result)
        if result.get("status")=="WAITING_VISUAL_REVIEW": break
    if failure is not None:
        failure_path = Path(args.output).expanduser().resolve() / "e2e_failure.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(failure_path, {
            "FINAL_E2E_STATUS": "FAIL",
            "INPUT_MODE": mode,
            "SCENE_PROMPT": prompt,
            "MOCK_FALLBACK_USED": False,
            "FAILED_STAGE": failure["stage"],
            "ERROR": failure["error"],
            "STAGE_RUN_RESULTS": results,
            "DOWNSTREAM_STAGES": {stage: "NOT_RUN" for stage in selected_stages[selected_stages.index(failure["stage"]) + 1:]},
        })
        print(json.dumps({"FINAL_E2E_STATUS": "FAIL", "failed_stage": failure["stage"],
                          "error": failure["error"], "output": str(failure_path)}, indent=2, ensure_ascii=False))
        return 1
    report=_finalize_report(args.config,args.output,mode,prompt,bool(seed_enabled),results,anchor_override) if should_finalize_report(results) else None
    if report is None:
        project=resolve_project_root(config_path, cfg)
        p=paths(cfg,Path(args.output).resolve(),project)
        report={"FINAL_OUTPUT":str(p["seedvr2_video"]) if seed_enabled and p["seedvr2_video"].is_file() else str(p["fa_video"]) if p["fa_video"].is_file() else None}
    print(json.dumps({"stages":results,"output":str(Path(args.output).resolve()),"input_mode":mode,
                      "seedvr2_enabled":bool(seed_enabled),"final_output":report.get("FINAL_OUTPUT")},
                     indent=2,ensure_ascii=False,default=str))
    return 0

if __name__=="__main__": raise SystemExit(main())

