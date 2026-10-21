"""Execute once and consume the predeclared controlled native CNS source."""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import yaml
from Libs.utils.model import controlled_cns_source as controlled
from Libs.utils.model import native_observable_source as observable
from Libs.utils.data.native_observable_profile import observable_profile, family_grid, quantity_support, profile_identity
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_tsv, read_tsv
from Libs.utils.model.microenvironment_teacher import write_microenvironment_zarr
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.data.joint_physical_parameters import parameter_declarations, parameter_schema_from_config, bind_candidate_parameters
from Libs.utils.data.native_joint_source_storage import write_joint_source_reference, read_joint_source_reference
from Libs.utils.data.native_field_normalization import fit_multigrid, bind_artifact, NativeNormalizationContext
from Libs.scripts.builders import build_native_diffusion_training_data as preparation
from Libs.exp.training.pretrain_microenvironment_world_model import (
    MicroenvironmentZarrDataset,MicroenvironmentTeacherFamilyDataset,collate_native_sources)
from Libs.exp.evaluation.evaluate_screening_system import _native_episode_identity,_native_validity_mask
from Libs.utils.model import native_scientific_source as scientific_source
from Libs.utils.model import native_romani_source as romani_source

SOURCE_FILE="Libs/utils/model/controlled_cns_source.py"
GIT_COMMAND = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false"]


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload,allow_nan=False,indent=2)+"\n")
    temp.replace(path)


class NativeStorageBudget:
    """Reserve one bounded complete group before writing; count only declared outputs."""
    def __init__(self, limit, group_limit, terminal_reserve=0, initial_bytes=0):
        self.limit, self.group_limit, self.terminal_reserve = int(limit), int(group_limit), int(terminal_reserve)
        self.used = int(initial_bytes)
        self.files = {}
        if min(self.limit, self.group_limit) <= 0 or self.terminal_reserve < 0:
            raise ValueError("Native storage byte ceilings must be positive")

    def reserve_group(self):
        if self.used + self.group_limit + self.terminal_reserve > self.limit:
            raise RuntimeError("Native storage budget cannot reserve the next complete group")

    @staticmethod
    def _file_sizes(path):
        """Stream retained files with one cached stat per ordinary entry.

        Scandir preserves unexpected-file accounting. Do not descend symlink
        directories, matching Path.rglob; charge regular-file symlink targets.
        """
        try:
            metadata = path.stat()
        except FileNotFoundError as error:
            raise FileNotFoundError("Declared native output is missing: " + str(path)) from error
        if stat.S_ISREG(metadata.st_mode):
            yield path, metadata.st_size
            return
        if not stat.S_ISDIR(metadata.st_mode):
            return
        pending = [path]
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(metadata.st_mode):
                        yield Path(entry.path), metadata.st_size
                    elif stat.S_ISLNK(metadata.st_mode):
                        try:
                            target = entry.stat(follow_symlinks=True)
                        except FileNotFoundError:
                            continue
                        if stat.S_ISREG(target.st_mode):
                            yield Path(entry.path), target.st_size

    def record(self, paths, *, complete_group=False):
        added = 0
        for path in paths:
            path = Path(path)
            for entry, size in self._file_sizes(path):
                added += size - self.files.get(entry, 0)
                self.files[entry] = size
        if complete_group and added > self.group_limit:
            raise RuntimeError("Complete group exceeded its reserved write ceiling")
        self.used += added
        if self.used + self.terminal_reserve > self.limit:
            raise RuntimeError("Native storage byte ceiling exceeded")


def planned_groups(source):
    groups=[]
    for condition in source["initial_conditions"]:
        for replicate in source["replicates"]:
            uid=source["source_uid"]+":"+condition+":replicate_"+str(replicate)
            groups.append({"group_uid":uid,"sample_uid":uid,"support_uid":uid+":support",
                "posterior_uid":uid+":predose-prior","particle_uids":[uid+":atom_"+str(j) for j in range(4)],
                "condition":condition,"replicate":replicate,"role":"development" if replicate==3 else "train"})
    return groups


class SourceLedger:
    """One durable pre-start record per prebound source operation; no replay."""
    def __init__(self,path,source,identity,clock=time.monotonic):
        self.path=path;self.clock=clock;self.start=clock();self.deadline=self.start+source["budget"]["seconds"]
        groups=planned_groups(source)
        cases=[{"case_uid":g["group_uid"]+":"+a["regimen_uid"],"group_index":gi,"action_index":ai,"status":"planned"}
               for gi,g in enumerate(groups) for ai,a in enumerate(source["actions"])]
        self.data={"source_identity":identity,"definition":source,"source_clock_seconds":0.,
            "source_status":"running","predictive_evaluation":"NOT_RUN","groups":groups,"cases":cases,
            "operations":{},"resource_limit":{"cpus":1,"memory_gib":2,"source_seconds":1800},
            "host":os.uname().nodename,"allocation":os.environ.get("SLURM_JOB_ID"),"step":os.environ.get("SLURM_STEP_ID")}
        for gi,group in enumerate(groups):self.data["operations"]["group:"+str(gi)]={"kind":"group","status":"planned"}
        for ci,case in enumerate(cases):
            for method in ("coarse","fine","reference"):
                self.data["operations"][f"integrate:{ci}:{method}"]={"kind":"integrations","status":"planned"}
            for atom in range(4):
                for kind in ("primary_pk_grids","precision_pk_grids"):
                    self.data["operations"][f"{kind}:{ci}:{atom}"]={"kind":kind,"status":"planned"}
        self.save()

    def save(self):
        self.data["source_clock_seconds"]=self.clock()-self.start
        write_json(self.path,self.data)

    def execute(self,key,function):
        entry=self.data["operations"][key]
        if entry["status"]!="planned":raise RuntimeError("Source operation already consumed: "+key)
        if self.clock()>self.deadline:raise TimeoutError("controlled source clock exhausted")
        entry.update(status="started",started_seconds=self.clock()-self.start);self.save()
        try:
            value=function()
        except BaseException as error:
            entry.update(status="failed",reason=type(error).__name__+": "+str(error));self.save();raise
        entry.update(status="complete",finished_seconds=self.clock()-self.start);self.save()
        return value


def bound_identity(source,config_path):
    commit=subprocess.check_output([*GIT_COMMAND,"rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    bound_paths=[ROOT/SOURCE_FILE,Path(__file__).resolve(),config_path.resolve()]
    for path in bound_paths:
        relative=path.relative_to(ROOT).as_posix()
        committed=subprocess.check_output([*GIT_COMMAND,"show",commit+":"+relative],cwd=ROOT)
        if committed!=path.read_bytes():raise ValueError("Source law/builder/config must be committed before draws: "+relative)
    source_bytes=(ROOT/SOURCE_FILE).read_bytes()
    return {"definition_sha256":digest(source),"source_code_sha256":hashlib.sha256(source_bytes).hexdigest(),
            "producer_commit":commit,"source_file":SOURCE_FILE,
            "builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_path":str(config_path),"config_file_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest()}


def verify_source_identity(receipt,source):
    identity=receipt["source_identity"]
    if identity["definition_sha256"]!=digest(source):raise ValueError("Source definition changed after generation")
    bound=subprocess.check_output(["git","show",identity["producer_commit"]+":"+identity["source_file"]],cwd=ROOT)
    if hashlib.sha256(bound).hexdigest()!=identity["source_code_sha256"]:raise ValueError("Bound source code identity mismatch")
    if hashlib.sha256((ROOT/identity["source_file"]).read_bytes()).hexdigest()!=identity["source_code_sha256"]:
        raise ValueError("Source law changed after generation")


def execute_source(config,config_path,*,integrator=controlled.integrate):
    source=controlled.definition(config);output=resolve_repo_path(source["output_root"])
    identity=bound_identity(source,config_path)
    output.mkdir(parents=True,exist_ok=False)
    local=Path(tempfile.mkdtemp(prefix="controlled_source_run_",dir=os.environ.get("TMPDIR","/tmp")))
    write_json(output/"source_reservation.json",{"source_identity":identity,"node_local_path":str(local),
        "host":os.uname().nodename,"status":"reserved_no_replay"})
    ledger=SourceLedger(local/"source_receipt.json",source,identity)
    scales=[source["reference_scales"][k] for k in source["grid"]["species_id"]]
    try:
        for gi,group in enumerate(ledger.data["groups"]):
            support=production=truth=None;programs=[[None,None] for _ in range(4)]
            for ai,action in enumerate(source["actions"]):
                ci=2*gi+ai;case=ledger.data["cases"][ci]
                if time.monotonic()>ledger.deadline:raise TimeoutError("controlled source clock exhausted")
                case.update(status="started",started_seconds=time.monotonic()-ledger.start);ledger.save()
                try:
                    if support is None:
                        support,production,truth=ledger.execute("group:"+str(gi),lambda:controlled.draw_world(source,group,gi))
                        write_json(local/f"group_{gi}_producer.json",{"hidden_truth_atom":truth,
                            "initial_seed":source["seed"]+10*gi,"latent_seed":production.latent_seed,"observation_seed":production.observation_seed})
                    pk_error=np.zeros(15);pk_serial=np.zeros(15)
                    for atom in range(4):
                        law=controlled.ControlledLaw(source,{k:v[0,atom].numpy() for k,v in support.factors.items()},
                                                    support.initial_field[0,atom].numpy())
                        pk=ledger.execute(f"primary_pk_grids:{ci}:{atom}",lambda:controlled.pk_values(source["grid"]["time_hours"],law.pk,action,source))
                        precise=ledger.execute(f"precision_pk_grids:{ci}:{atom}",lambda:controlled.pk_values(source["grid"]["time_hours"],law.pk,action,source,high_precision=True))
                        for si,pi in ((0,0),(1,2)):
                            pk_error[si]=max(pk_error[si],float(np.max(np.abs(pk[pi]-precise[pi]))))
                            pk_serial[si]=max(pk_serial[si],float(np.max(np.abs(pk[pi]-pk[pi].astype(np.float32).astype(float)))))
                        programs[atom][ai]=controlled.compiled_particle_program(source,group,support,atom,action,pk,law.pk,identity["source_code_sha256"])
                    law=controlled.ControlledLaw(source,{k:v[0,truth].numpy() for k,v in support.factors.items()},
                                                support.initial_field[0,truth].numpy())
                    results={};failures=[]
                    for label,method,step in (("coarse","RK4",1.),("fine","RK4",.5),("reference","DOP853",.25)):
                        try:
                            value=ledger.execute(f"integrate:{ci}:{label}",lambda:integrator(law,action,method,step,ledger.deadline))
                            results[label]=value
                            np.savez_compressed(local/f"case_{ci}_{label}.npz",**value)
                        except TimeoutError:raise
                        except (ValueError,RuntimeError,FloatingPointError) as error:failures.append(str(error))
                    if len(results)==3:
                        summary=controlled.numerical_summary(results["coarse"],results["fine"],results["reference"],
                                                            scales,pk_error,pk_serial,source["numerics"])
                        case.update(status="complete",numerical=summary,raw_reference=f"case_{ci}_reference.npz")
                    else:case.update(status="failed",reason="; ".join(failures))
                except TimeoutError:raise
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    case.update(status="failed",reason=type(error).__name__+": "+str(error))
                case["finished_seconds"]=time.monotonic()-ledger.start;ledger.save()
            if support is not None and all(v is not None for row in programs for v in row):
                raw=controlled.joint_source(source,group,support,production,programs)
                write_json(local/f"group_{gi}_conditioning.json",raw)
            else:group["conditioning_status"]="incomplete"
        ledger.data["source_status"]="complete"
    except BaseException as error:
        ledger.data.update(source_status="failed",reason=type(error).__name__+": "+str(error))
    finally:
        for case in ledger.data["cases"]:
            if case["status"]=="planned":case["status"]="missing"
            elif case["status"]=="started":case["status"]="failed";case["reason"]="source execution interrupted"
        ledger.data["source_peak_rss_kib"]=max(ledger.data.get("source_peak_rss_kib",0),
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ledger.save()
        # Transfer only this run's declared flat source outputs, once per file.
        files=["source_receipt.json"]+[f"group_{i}_{kind}.json" for i in range(8) for kind in ("producer","conditioning")]
        files += [f"case_{i}_{kind}.npz" for i in range(16) for kind in ("coarse","fine","reference")]
        for name in files:
            if (local/name).exists():shutil.copyfile(local/name,output/name)
        shutil.rmtree(local)
    return output


def call_preparation(arguments):
    previous=sys.argv;captured=io.StringIO()
    try:
        sys.argv=["build_native_diffusion_training_data.py",*arguments]
        with contextlib.redirect_stdout(captured):
            code=preparation.main()
        if code!=0:raise RuntimeError("Native preparation failed")
    finally:sys.argv=previous
    return {"argv":arguments,"stdout":captured.getvalue(),"exit_code":code}


def build_native_source_store(config,config_path):
    source=controlled.definition(config);root=resolve_repo_path(source["output_root"])
    receipt=json.loads((root/"source_receipt.json").read_text());verify_source_identity(receipt,source)
    report={"predictive_evaluation":"NOT_RUN","declared_cases":16,"source_status":receipt["source_status"],
            "cases":[],"prepared_roles":{},"source_identity":receipt["source_identity"]}
    qc=[];accepted=[];joint_sources=[];trajectories=[];uncertainty=[]
    store=root/"native_source.zarr";registry_path=root/"native_registry.json"
    source_row={"source_uid":source["source_uid"],"public":True,"protected":False,"license":source["source_license"],
        "release":"prospective_controlled_v1","snapshot_ref":source["snapshot_ref"],
        "split_evidence_ref":str(root/"source_receipt.json"),"unit_evidence_ref":str(config_path),
        "source_kind":"controlled_simulated","exact_zero_quantities":source["exact_zero_quantities"],
        "predictive_evaluation_units":{"episode_field":"sample_uid","group_field":"family_uid",
            "group_kind":"action_family","independent_episode_units":True,
            "evidence_ref":str(root/"source_receipt.json")}}
    for ci,case in enumerate(receipt["cases"]):
        group=receipt["groups"][case["group_index"]];action=source["actions"][case["action_index"]]
        conditioning=root/f'group_{case["group_index"]}_conditioning.json'
        reasons=list(case.get("numerical",{}).get("reasons",[]))
        if case["status"]!="complete":reasons.append(case.get("reason",case["status"]))
        if not conditioning.exists():reasons.append("complete independent conditioning missing")
        accepted_case=not reasons
        status={"case_uid":case["case_uid"],"sample_uid":group["sample_uid"],"role":group["role"],
                "source_status":case["status"],"native_status":"candidate" if accepted_case else "rejected","reasons":reasons}
        report["cases"].append(status)
        quality=case.get("numerical",{})
        q={"trajectory_uid":case["case_uid"],"source_uid":source["source_uid"],"sample_uid":group["sample_uid"],
            "family_uid":group["group_uid"],"qc_grade":"B" if accepted_case else "D",
            "qc_use":"main_pretraining" if accepted_case else "reject","sample_weight":.5 if accepted_case else 0.,
            "split_name":group["role"],"source_holdout_role":group["role"],
            "teacher_confidence":quality.get("teacher_confidence",0.)}
        qc.append(q)
        if not accepted_case:continue
        raw=json.loads(conditioning.read_text())
        native.parse_joint_source(raw)
        if raw["source_posterior_uid"] not in {s["source_posterior_uid"] for s in joint_sources}:joint_sources.append(raw)
        program=raw["particle_programs"]["records"][0][0][case["action_index"]]
        target=np.load(root/case["raw_reference"])["trajectory"].astype(np.float32)
        teacher={**q,**{k:program[k] for k in ("regimen_uid","action_uid","target_program_uid","program_uid","posterior_uid")},
            "content_hash":hashlib.sha256(target.tobytes()).hexdigest(),"trajectory_store_path":str(store),
            "store_format":"zarr_v2","trajectory_index":len(accepted),"teacher_family":"full_microenvironment_teacher",
            "qc_catalog_ref":str(root/"simulation_qc_catalog.tsv"),"source_snapshot_ref":source["snapshot_ref"],
            "uncertainty_origin":"independent_supplied","numerical_evidence":quality,
            "validity_mask":{"kind":"all_valid","evidence_ref":str(root/"source_receipt.json")},
            "source_case_index":ci}
        accepted.append(teacher);trajectories.append(target);uncertainty.append(quality["uncertainty"])
    write_tsv(root/"simulation_qc_catalog.tsv",list(dict.fromkeys(key for row in qc for key in row)),qc)
    joined={row["trajectory_uid"]:row for row in read_tsv(root/"simulation_qc_catalog.tsv")}
    for row in accepted:
        q=joined[row["trajectory_uid"]]
        for key in ("qc_grade","qc_use","split_name","source_holdout_role"):row[key]=q[key]
        for key in ("sample_weight","teacher_confidence"):row[key]=float(q[key])
    registry={"schema_version":native.SCHEMA_VERSION,"grid":controlled.source_grid(source),"sources":[source_row],
        "posterior_family":"finite_joint_particles","joint_sources":joint_sources,"teachers":accepted,"programs":[],"observations":[]}
    report["candidate_count"]=len(accepted)
    if not accepted:
        report.update(status="incomplete",admitted_count=0)
        write_json(root/"native_consumption.json",report);return report
    # A source holds both roles. It is not multiple independent biological sources.
    train=[row for row in accepted if row["split_name"]=="train"]
    development=[row for row in accepted if row["split_name"]=="development"]
    report["role_overlap"] = {key:sorted({row[key] for row in train}&{row[key] for row in development})
                              for key in ("trajectory_uid","content_hash","family_uid","source_uid")}
    write_json(root/"native_consumption.json",report)
    for key in ("trajectory_uid","content_hash","family_uid"):
        if report["role_overlap"][key]:raise ValueError("Role leakage: "+key)
    arrays={"trajectory":np.stack(trajectories),"uncertainty":np.asarray(uncertainty,dtype=np.float32),
            "teacher_confidence":np.asarray([t["teacher_confidence"] for t in accepted],dtype=np.float32)}
    write_json(registry_path,registry)
    effective=copy.deepcopy(config);effective["native_diffusion_data"]["registry_path"]=str(registry_path)
    local_config=root/"native_config.yaml";local_config.write_text(yaml.safe_dump(effective,sort_keys=False))
    if store.exists():raise FileExistsError("Consumed source store exists; no overwrite")
    write_microenvironment_zarr(store,arrays["trajectory"],arrays["uncertainty"],effective,
        {"source_kind":"controlled_simulated","source_identity":receipt["source_identity"]},
        native_payload={"registry":registry,"arrays":arrays})
    report["native_writer_rows"]=len(accepted)
    write_json(root/"native_consumption.json",report)
    return report


def consume_dataset(dataset, expected, config, *, normalization_context=None):
    """Retain both actions and all particles, releasing each complete group."""
    records=dataset.native_records
    if [r["teacher"]["trajectory_uid"] for r in records]!=expected:
        raise ValueError("Dataset identity/order mismatch")
    consumed=[];groups=[]
    for sample in dict.fromkeys(r["teacher"]["sample_uid"] for r in records):
        indices=[i for i,r in enumerate(records) if r["teacher"]["sample_uid"]==sample]
        current=[records[i] for i in indices]
        _native_episode_identity(current,dataset.native_registry,
            config["paper_grade_full_suite"]["native_rollout_evaluation"],config)
        _native_validity_mask(current,dataset.native_registry["grid"])
        conditioning,bindings=native.load_records(dataset.native_registry,current,config,include_teacher=False,
                                                normalization_context=normalization_context)
        if conditioning:raise ValueError("Joint conditioning unexpectedly read teacher arrays")
        del conditioning,bindings
        items=dataset.__getitems__(indices)
        identities=[row["trajectory_uid"] for row in items]
        if identities!=[expected[i] for i in indices]:raise ValueError("Dataset group identity/order mismatch")
        if len(collate_native_sources(items)["joint_rows"])!=len(indices):raise ValueError("Native collate row loss")
        if not all(torch.isfinite(item["trajectory"]).all() and torch.isfinite(item["uncertainty"]).all() for item in items):
            raise ValueError("Nonfinite native dataset")
        consumed.extend(identities);groups.append(sample)
        del items,current
    if consumed!=expected:raise ValueError("Complete-group consumption lost or reordered rows")
    return {"rows":len(consumed),"groups":groups}


def reuse_completed_store(root, receipt):
    """Reuse the one completed writer result, rejecting stale source bindings."""
    report=json.loads((root/"native_consumption.json").read_text())
    if (receipt["source_status"]!="complete" or report.get("status")!="complete"
            or report.get("admitted_count")!=len(receipt["cases"])
            or report.get("source_identity")!=receipt["source_identity"]):
        raise ValueError("Reuse requires the completed matching source/native receipt")
    attributes=json.loads((root/"native_source.zarr"/".zattrs").read_text())
    if attributes.get("source_identity")!=receipt["source_identity"]:
        raise ValueError("Reused native store source identity mismatch")
    registry=json.loads((root/"native_registry.json").read_text())
    if attributes["native_diffusion"]!=registry:
        raise ValueError("Reused native store registry mismatch")
    qc={row["trajectory_uid"]:row for row in read_tsv(root/"simulation_qc_catalog.tsv")}
    expected=[row["case_uid"] for row in receipt["cases"]]
    if [row["trajectory_uid"] for row in registry["teachers"]]!=expected:
        raise ValueError("Reused source case denominator/order mismatch")
    for row in registry["teachers"]:
        joined=qc[row["trajectory_uid"]]
        for key in ("qc_grade","qc_use","split_name","source_holdout_role"):
            if row[key]!=joined[key]:raise ValueError("Reused QC join mismatch")
        for key in ("sample_weight","teacher_confidence"):
            if row[key]!=float(joined[key]):raise ValueError("Reused QC numerical join mismatch")
    report=copy.deepcopy(report)
    report["prepared_roles"]={}
    report["preparation_commands"]=[]
    report["reused_completed_outputs"]=True
    for row in report["cases"]:row["native_status"]="candidate"
    return report


def consume_source(config,config_path,*,reuse_completed=False):
    source=controlled.definition(config);root=resolve_repo_path(source["output_root"])
    receipt_path=root/"source_receipt.json"
    source_receipt_bytes=receipt_path.read_bytes()
    receipt=json.loads(source_receipt_bytes);verify_source_identity(receipt,source)
    report_path=root/("native_consumption_recheck.json" if reuse_completed else "native_consumption.json")
    if reuse_completed:
        if report_path.exists():raise FileExistsError("Consumption recheck receipt already exists")
        report=reuse_completed_store(root,receipt)
    else:
        report=build_native_source_store(config,config_path)
        if not report.get("candidate_count"):return report
    gc.collect()
    effective=load_config(str(root/"native_config.yaml"))
    normalization=root/"train_normalization.json"
    if not reuse_completed:
        report["preparation_commands"]=[call_preparation(["--config",str(root/"native_config.yaml"),
            "--split","train","--max-rows","16","--fit-field-normalization","--output",str(normalization)])]
    artifact=json.loads(normalization.read_text())
    expected_train=[row["case_uid"] for row in report["cases"] if row["role"]=="train" and row["native_status"]=="candidate"]
    if artifact.get("fit_role")!="train" or [r["trajectory_uid"] for r in artifact["training_records"]]!=expected_train:
        raise ValueError("Training normalization source/role identities mismatch")
    for role in ("train","development"):
        role_store=root/(role+"_normalized.zarr")
        if not reuse_completed:
            report["preparation_commands"].append(call_preparation(["--config",str(root/"native_config.yaml"),
                "--split",role,"--max-rows","16","--normalization-artifact",str(normalization),"--output",str(role_store)]))
        elif not (role_store/".zattrs").is_file():
            raise ValueError("Completed normalized role store missing")
        role_config=copy.deepcopy(effective);role_config["native_diffusion_data"]["registry_path"]=str(role_store/".zattrs")
        expected=[r["case_uid"] for r in report["cases"] if r["role"]==role and r["native_status"]=="candidate"]
        checks={}
        for kind in ("direct","family"):
            dataset=(MicroenvironmentZarrDataset(role_store,role_config,split_name=role) if kind=="direct"
                     else MicroenvironmentTeacherFamilyDataset(role_config,role))
            norm=dataset.native_registry["grid"]["normalization"]
            if norm.get("artifact_ref")!=str(normalization) or norm.get("fit_role")!="train":
                raise ValueError("Role normalization is not the shared train artifact")
            checks[kind]=consume_dataset(dataset,expected,role_config)
            del dataset
            gc.collect()
            print(json.dumps({"phase":"native_dataset_consumed","role":role,"dataset":kind,
                "rows":checks[kind]["rows"],"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
        report["prepared_roles"][role]={"rows":len(expected),"trajectory_uids":expected,"both_datasets":"consumed",
            "dataset_checks":checks,"collate":"joint_rows","evaluator_source_identity_and_mask":"valid",
            "teacher_outcome_reads_during_conditioning":False,"normalization_artifact":str(normalization)}
        for row in report["cases"]:
            if row["native_status"]=="candidate" and row["role"]==role:row["native_status"]="admitted"
        write_json(report_path,report)
        del role_config,checks
        gc.collect()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    within_memory=peak<=receipt["resource_limit"]["memory_gib"]*1024**2
    unchanged=receipt_path.read_bytes()==source_receipt_bytes
    if not unchanged:raise ValueError("Source receipt changed during consumption")
    report.update(admitted_count=sum(row["native_status"]=="admitted" for row in report["cases"]),
        status="complete" if all(row["native_status"]=="admitted" for row in report["cases"]) and within_memory else "incomplete",
        source_count=1,source_holdout_coverage=False,consumer_peak_rss_kib=peak,
        consumer_resource_status="within_limit" if within_memory else "memory_limit_exceeded",
        source_receipt_unchanged=unchanged,
        source_receipt_sha256=hashlib.sha256(source_receipt_bytes).hexdigest(),
        consumer_commit=subprocess.check_output([*GIT_COMMAND,"rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        consumer_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(report_path,report)
    return report



def observable_source_api(config):
    if config["native_observable_bridge"].get("scientific_profile", False):
        return scientific_source
    return observable


class ObservableSourceLedger(SourceLedger):
    """The fixed family/group/case ledger; a started operation is never replayed."""
    def __init__(self, path, config, identity, clock=time.monotonic, *, persist_initial=True):
        observable = observable_source_api(config)
        settings=config["native_observable_bridge"]
        self.group_snapshot_mode = bool(settings.get("scientific_profile") and
            config["native_diffusion_data"].get("group_storage", {}).get("kind") == "gzip_json_complete_groups")
        self.active_group = None
        self.group_receipt_names = []
        self.path=path; self.clock=clock; self.start=clock(); self.deadline=self.start+settings["budget"]["seconds"]
        groups=observable.group_definitions(config)
        cases=[{"case_uid":group["group_uid"]+":case:"+str(ai),"group_index":gi,"action_index":ai,"status":"planned"}
               for gi,group in enumerate(groups) for ai in range(scientific_source.action_count(config,group))]
        self.data={"source_identity":identity,"definition":settings,"source_clock_seconds":0.,
            "source_status":"running","predictive_evaluation":"NOT_RUN","groups":groups,"cases":cases,"operations":{},
            "resource_limit":{"cpus":settings["budget"]["cpus"],"memory_gib":settings["budget"]["memory_gib"],
                              "source_seconds":settings["budget"]["seconds"]},
            "host":os.uname().nodename,"allocation":os.environ.get("SLURM_JOB_ID"),"step":os.environ.get("SLURM_STEP_ID")}
        operations=self.data["operations"]
        for gi in range(len(groups)):
            operations["group:"+str(gi)]={"kind":"groups","status":"planned"}
        romani_groups = [gi for gi, group in enumerate(groups) if group.get("source_law") == "romani_vm_source"]
        if romani_groups:
            operations["source_prepare:"+str(romani_groups[0])] = {"kind":"source_prepare","status":"planned"}
            for gi in romani_groups:
                for atom in range(scientific_source.prior_particle_count(config)):
                    operations[f"parameter_binding:{gi}:{atom}"] = {"kind":"parameter_binding","status":"planned"}
        for ci,case in enumerate(cases):
            if case["group_index"] in romani_groups:
                operations["physical_refinement:"+str(ci)] = {"kind":"physical_refinement","status":"planned"}
            if not scientific_source.is_long_cns(config, groups[case["group_index"]]["family_uid"]):
                for kind in ("physical","physical_precision"):
                    operations[kind+":"+str(ci)]={"kind":kind,"status":"planned"}
            for label in ("coarse","fine","reference"):
                operations[f"cns:{ci}:{label}"]={"kind":"cns_integrations","status":"planned"}
            for atom in range(scientific_source.prior_particle_count(config)):
                operations[f"program:{ci}:{atom}"]={"kind":"programs","status":"planned"}
                if (groups[case["group_index"]]["family_uid"]==observable.MOLECULAR
                        or scientific_source.is_long_cns(config, groups[case["group_index"]]["family_uid"])):
                    operations[f"chemical_precision:{ci}:{atom}"]={"kind":"chemical_precision","status":"planned"}
        expected_counts = scientific_source.planned_operation_counts(config)
        if any(settings["budget"].get(key) != value for key,value in expected_counts.items()):
            raise ValueError("Declared observable operation counts differ from the complete prior/action design")
        if (len(groups)!=settings["budget"]["groups"] or len(cases)!=settings["budget"]["cases"]
                or len(operations)!=settings["budget"]["operations"]):
            raise ValueError("Declared observable operation/group/case denominator mismatch")
        self.group_case_indices = {gi: [] for gi in range(len(groups))}
        self.group_operation_keys = {gi: [] for gi in range(len(groups))}
        for ci, case in enumerate(cases):
            self.group_case_indices[case["group_index"]].append(ci)
        for key in operations:
            self.group_operation_keys[self.operation_group_index(key)].append(key)
        if persist_initial:
            self.save()

    def save(self):
        if not self.group_snapshot_mode or self.data["source_status"] != "running":
            return super().save()
        self.data["source_clock_seconds"] = self.clock() - self.start
        if self.active_group is None:
            # A running root is not an execution receipt. Operation starts are
            # persisted in the active complete-group snapshot before execution.
            initial = dict(self.data, operations={}, planned_operation_count=len(self.data["operations"]),
                           operation_snapshot_status="group_snapshots_active")
            write_json(self.path, initial)
            return
        gi = self.active_group
        name = f"group_{gi}_receipt.json"
        if name not in self.group_receipt_names:
            self.group_receipt_names.append(name)
        snapshot = {"source_identity": self.data["source_identity"], "group_index": gi,
                    "group": self.data["groups"][gi],
                    "cases": [self.data["cases"][ci] for ci in self.group_case_indices[gi]],
                    "operations": {key: self.data["operations"][key] for key in self.group_operation_keys[gi]},
                    "source_clock_seconds": self.data["source_clock_seconds"],
                    "interrupted_started_operation": "consumed_outcome_unknown_no_replay"}
        write_json(self.path.parent/name, snapshot)

    def operation_group_index(self, key):
        prefix, number, *_ = key.split(":")
        return (int(number) if prefix in {"group", "source_prepare", "parameter_binding"}
                else self.data["cases"][int(number)]["group_index"])

    def execute(self, key, function):
        if self.group_snapshot_mode:
            self.active_group = self.operation_group_index(key)
        return super().execute(key, function)


def observable_code_identity(settings, identities):
    """Prospective source IDs bind the consumed frozen producer dependency set."""
    source_file = ("Libs/utils/model/native_scientific_source.py" if settings.get("scientific_profile")
                   else "Libs/utils/model/native_observable_source.py")
    code_hash = (digest({path: identities[path] for path in settings["producer_freeze_paths"]})
                 if settings.get("scientific_profile") else identities[source_file])
    return source_file, code_hash


def observable_identity(config, config_path):
    settings=config["native_observable_bridge"]
    commit=subprocess.check_output([*GIT_COMMAND,"rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    identities={}
    for relative in settings["freeze_paths"]:
        path=ROOT/relative
        committed=subprocess.check_output([*GIT_COMMAND,"show",commit+":"+relative],cwd=ROOT)
        if committed!=path.read_bytes():
            raise ValueError("Observable producer/config/schema/tests must be committed before production: "+relative)
        identities[relative]=hashlib.sha256(committed).hexdigest()
    relative=config_path.resolve().relative_to(ROOT).as_posix()
    if relative not in identities:
        raise ValueError("Production entry config is outside the source freeze")
    source_file, code_hash = observable_code_identity(settings, identities)
    return {"producer_commit":commit,"definition_sha256":digest({"profile":settings,"background":config["controlled_cns_source"]}),
            "bound_files":identities, "source_file":source_file, "source_code_sha256":code_hash}



def verify_observable_source(receipt, config):
    observable = observable_source_api(config)
    identity=receipt["source_identity"];settings=config["native_observable_bridge"]
    source_file, code_hash = observable_code_identity(settings, identity["bound_files"])
    if identity["source_file"] != source_file or identity["source_code_sha256"] != code_hash:
        raise ValueError("Observable source code identity differs from its frozen producer dependencies")
    if identity["definition_sha256"]!=digest({"profile":settings,"background":config["controlled_cns_source"]}):
        raise ValueError("Observable source definition changed after generation")
    declared_groups = observable.group_definitions(config)
    actual_groups = receipt.get("groups")
    if settings.get("scientific_profile"):
        actual_groups = [{key: value for key, value in group.items() if key != "conditioning_reference"}
                         for group in receipt.get("groups", [])]
        if receipt.get("source_status") == "complete":
            for gi, group in enumerate(receipt["groups"]):
                reference = group.get("conditioning_reference", {})
                if reference.get("source_posterior_uid") != group["posterior_uid"]:
                    raise ValueError("Scientific complete-group reference identity differs")
                expected = resolve_repo_path(settings["output_root"])/f"group_{gi}_conditioning.json.gz"
                if resolve_repo_path(reference.get("path", "")) != expected:
                    raise ValueError("Scientific complete-group reference is outside its retained output path")
    if (actual_groups!=declared_groups
            or len(receipt.get("cases",[]))!=settings["budget"]["cases"]
            or len(receipt.get("operations",{}))!=settings["budget"]["operations"]):
        raise ValueError("Observable source receipt denominator or declared group identity differs")
    if receipt.get("source_status")=="complete" and any(op.get("status")!="complete" for op in receipt["operations"].values()):
        raise ValueError("Completed observable source has uncompleted operations")
    for relative in settings["producer_freeze_paths"]:
        expected=identity["bound_files"][relative]
        committed=subprocess.check_output([*GIT_COMMAND,"show",identity["producer_commit"]+":"+relative],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest()!=expected or hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError("Frozen observable producer changed: "+relative)


def execute_long_cns_case(config, ledger, group, generated, action, records, action_index,
                          case_index, identity, local, declared):
    """Use the actual CNS PK event compiler and source solver on their28-time grid."""
    source = scientific_source.long_cns_definition(config)
    source["pk"]["total_dose_mg_per_kg"] *= action.get("total_dose_multiplier",1.)
    pk_error, pk_serialization = np.zeros(15), np.zeros(15)
    for atom in range(scientific_source.prior_particle_count(config)):
        law = scientific_source.long_cns_law(config, group, generated["support"], atom, action)
        def program():
            pk = controlled.pk_values(source["grid"]["time_hours"], law.pk, action, source)
            record = controlled.compiled_particle_program(source,group,generated["support"],atom,
                action,pk,law.pk,identity["source_code_sha256"])
            return pk, record
        pk, records[atom][action_index] = ledger.execute(f"program:{case_index}:{atom}", program)
        reference = ledger.execute(f"chemical_precision:{case_index}:{atom}",
            lambda:controlled.pk_values(source["grid"]["time_hours"],law.pk,action,source,high_precision=True))
        for species, component in ((0,0),(1,2)):
            pk_error[species] = max(pk_error[species],float(np.abs(pk[component]-reference[component]).max()))
            pk_serialization[species] = max(pk_serialization[species],
                float(np.abs(pk[component]-pk[component].astype(np.float32).astype(float)).max()))
    law = scientific_source.long_cns_law(config,group,generated["support"],generated["hidden_index"],action)
    numerics = source["numerics"]
    steps = [*numerics["rk4_steps_hours"],numerics["reference_max_step_hours"]]
    results = {}
    for label,method,step in zip(("coarse","fine","reference"),("RK4","RK4","DOP853"),steps):
        results[label] = ledger.execute(f"cns:{case_index}:{label}",
            lambda:scientific_source.integrate_long_reference(law,action,method,step,ledger.deadline))
    quality = scientific_source.long_cns_numerical_summary(config,results,pk_error,pk_serialization)
    for label,result in results.items():
        # Padding carries only missing coordinates; quantity_support withholds all11.
        result["trajectory"] = np.pad(result["trajectory"],((0,0),(0,0),(0,0),(0,11)))
        name = f"case_{case_index}_{label}.npz"
        np.savez_compressed(local/name,**result)
        declared.append(name)
    return quality


def observable_source_roles(config):
    return list(dict.fromkeys(group["role"] for group in observable_source_api(config).group_definitions(config)))



def source_group_limit(config):
    """A finite complete-group boundary; legacy source invocations remain whole."""
    storage = config.get("native_diffusion_data", {}).get("group_storage", {})
    limit = storage.get("maximum_groups_per_invocation")
    if limit is not None and (type(limit) is not int or limit < 1
            or not config["native_observable_bridge"].get("scientific_profile")
            or storage.get("kind") != "gzip_json_complete_groups"):
        raise ValueError("Bounded source execution requires a positive scientific complete-group limit")
    return limit


def restore_source_prefix(ledger, previous):
    """Continue only after a normally completed group boundary, never a failed attempt."""
    start = previous.get("next_group_index")
    if (previous.get("source_status") != "partial" or type(start) is not int
            or not 0 < start < len(ledger.data["groups"])
            or previous.get("source_identity") != ledger.data["source_identity"]
            or previous.get("resource_limit") != ledger.data["resource_limit"]
            or digest(previous.get("definition")) != digest(ledger.data["definition"])
            or len(previous.get("cases",[])) != len(ledger.data["cases"])
            or len(previous.get("groups",[])) != len(ledger.data["groups"])
            or set(previous.get("operations", {})) != set(ledger.data["operations"])):
        raise ValueError("Source continuation requires its exact completed prefix")
    for ci, case in enumerate(previous["cases"]):
        expected = ledger.data["cases"][ci]
        if any(case[key] != expected[key] for key in ("case_uid", "group_index", "action_index")):
            raise ValueError("Source continuation changed a declared case")
        if case["status"] != ("complete" if case["group_index"] < start else "planned"):
            raise ValueError("Failed, started or missing source cases cannot be replayed")
    for key, operation in previous["operations"].items():
        expected = ledger.data["operations"][key]
        if (operation["kind"] != expected["kind"] or
                operation["status"] != ("complete" if ledger.operation_group_index(key) < start else "planned")):
            raise ValueError("Source continuation cannot replay an attempted or unresolved operation")
    elapsed = previous["source_clock_seconds"]
    if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Source continuation has invalid cumulative elapsed time")
    ledger.data = copy.deepcopy(previous)
    ledger.data["source_status"] = "running"
    ledger.start -= elapsed
    ledger.deadline = ledger.start + ledger.data["resource_limit"]["source_seconds"]
    if ledger.clock() >= ledger.deadline:
        raise TimeoutError("Cumulative controlled source clock exhausted")
    return start


def stage_source_group(local, output, names, budget, staged, retained_bytes, storage):
    """Copy declared immutable group outputs once, then release their local space."""
    paths = [local/name for name in dict.fromkeys(names)]
    budget.record(paths, complete_group=True)
    added = sum(path.stat().st_size for path in paths)
    if retained_bytes + added + budget.terminal_reserve > storage["retained_max_bytes"]:
        raise RuntimeError("Retained source budget cannot accept this complete group")
    for path in paths:
        destination = output/path.name
        if destination.exists():
            raise FileExistsError("Source output already exists; no overwrite: "+str(destination))
        shutil.copyfile(path, destination)
        staged.add(path.name)
    for path in paths:
        budget.used -= budget.files.pop(path)
        path.unlink()
    return retained_bytes + added



def stage_remaining_source_outputs(local, output, names, staged):
    """Preserve raw conflicts locally; only the running checkpoint is mutable."""
    conflicts=[]
    for name in dict.fromkeys(names):
        path=local/name
        if name in staged or not path.is_file():
            continue
        destination=output/name
        if destination.exists() and name!="source_receipt.json":
            conflicts.append({"local_path":str(path),"existing_path":str(destination)})
            continue
        shutil.copyfile(path,destination)
    return conflicts


def execute_observable_source(config, config_path, *, continue_source=False):
    observable = observable_source_api(config)
    settings=config["native_observable_bridge"]
    if settings.get("scientific_profile") and settings.get("source_contract_status") != "frozen":
        raise ValueError("Scientific source production requires its complete frozen corpus protocol")
    identity=observable_identity(config,config_path)
    output=resolve_repo_path(settings["output_root"])
    limit=source_group_limit(config)
    previous=None
    if continue_source:
        if limit is None:
            raise ValueError("Legacy source execution has no continuation contract")
        previous=json.loads((output/"source_receipt.json").read_text())
        verify_observable_source(previous,config)
        chunks=previous.get("completed_chunks",[])
        if previous.get("source_status")!="partial" or not chunks:
            raise ValueError("Only a normally completed source chunk can be continued")
        marker=json.loads((output/f'source_chunk_{chunks[-1]["first_group_index"]}.json').read_text())
        if (marker["status"]!="complete" or marker["source_identity"]!=identity
                or marker["source_status"]!="partial"
                or marker["stop_group_index"]!=previous["next_group_index"]
                or marker["source_clock_seconds"]!=previous["source_clock_seconds"]):
            raise ValueError("Source chunk outcome is unknown or its checkpoint changed")
        reservation=json.loads((output/"source_reservation.json").read_text())
        local=Path(reservation["node_local_path"])
        if (reservation["source_identity"] != identity or reservation["host"] != os.uname().nodename
                or not local.is_dir() or not local.name.startswith("observable_source_")):
            raise ValueError("Source continuation needs its original owned node-local workspace")
    else:
        output.mkdir(parents=True,exist_ok=False)
    with contextlib.ExitStack() as stack:
        if previous is None:
            if limit is None:
                local=Path(stack.enter_context(tempfile.TemporaryDirectory(
                    prefix="observable_source_",dir=os.environ.get("TMPDIR","/tmp"))))
            else:
                local=Path(tempfile.mkdtemp(prefix="observable_source_",dir=os.environ.get("TMPDIR","/tmp")))
            write_json(output/"source_reservation.json",{"source_identity":identity,"node_local_path":str(local),
                "host":os.uname().nodename,"status":"reserved_no_replay"})
        ledger=ObservableSourceLedger(local/"source_receipt.json",config,identity,persist_initial=limit is None)
        start=restore_source_prefix(ledger,previous) if previous is not None else 0
        stop=min(len(ledger.data["groups"]),start+limit) if limit is not None else len(ledger.data["groups"])
        chunk_path=output/f"source_chunk_{start}.json"
        if limit is not None:
            with chunk_path.open("x") as stream:
                json.dump({"source_identity":identity,"first_group_index":start,"stop_group_index":stop,
                    "status":"started_outcome_unknown_no_replay","host":os.uname().nodename,
                    "allocation":os.environ.get("SLURM_JOB_ID")},stream)
            ledger.save()
        declared=["source_receipt.json"]
        staged=set()
        retained_bytes=ledger.data.get("resource_bytes",{}).get("source_files_bytes",0)
        storage = config["native_diffusion_data"].get("group_storage", {})
        byte_budget = (NativeStorageBudget(storage["scratch_max_bytes"], storage["maximum_group_write_bytes"],
                                           storage["maximum_terminal_receipt_bytes"])
                       if settings.get("scientific_profile") and storage else None)
        try:
            global_config = config
            romani_runtime = None
            case_cursor = ledger.group_case_indices[start][0]
            for gi in range(start,stop):
                group=ledger.data["groups"][gi]
                if byte_budget:
                    byte_budget.reserve_group()
                    if (limit is not None and retained_bytes+byte_budget.group_limit+
                            byte_budget.terminal_reserve>storage["retained_max_bytes"]):
                        raise RuntimeError("Retained source budget cannot reserve the next complete group")
                first_declared = len(declared)
                config = (scientific_source.scoped_group_config(global_config,group)
                          if settings.get("scientific_profile") else global_config)
                initializer = None
                if group.get("source_law") == "romani_vm_source":
                    if romani_runtime is None:
                        def prepare():
                            work = local/"romani_source_runtime"
                            work.mkdir()
                            model = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]["modeldb"]
                            prepared = romani_source.stage_source(config, resolve_repo_path(model["extracted_root"]), work)
                            write_json(local/"romani_source_preparation.json", prepared)
                            declared.append("romani_source_preparation.json")
                            return romani_source.RomaniCell(config, prepared)
                        preparation_keys=[key for key,value in ledger.data["operations"].items()
                                          if value["kind"]=="source_prepare"]
                        if any(ledger.data["operations"][key]["status"]=="complete" for key in preparation_keys):
                            prepared=json.loads((output/"romani_source_preparation.json").read_text())
                            romani_runtime=romani_source.RomaniCell(config,prepared)
                            ledger.data.setdefault("reused_runtime_loads",[]).append({
                                "allocation":os.environ.get("SLURM_JOB_ID"),"group_index":gi,
                                "source_preparation_replayed":False,
                                "hoc_startup_initializations":romani_runtime.initialization_count})
                        else:
                            romani_runtime = ledger.execute("source_prepare:"+str(gi), prepare)
                    def initializer(candidates):
                        return [ledger.execute(f"parameter_binding:{gi}:{atom}",
                            lambda parameters=parameters:romani_runtime.initialize(parameters))
                            for atom,parameters in enumerate(candidates)]
                generated=ledger.execute("group:"+str(gi),lambda:observable.draw_group(config,group,gi,physical_initializer=initializer)
                    if initializer is not None else observable.draw_group(config,group,gi))
                producer_name=f"group_{gi}_producer.json"
                write_json(local/producer_name,{"hidden_index":generated["hidden_index"],"parameters":generated["parameters"],
                    "rng_streams":generated["rng_streams"],
                    **({"parameter_readbacks":generated["parameter_readbacks"]} if "parameter_readbacks" in generated else {})})
                declared.append(producer_name)
                actions = scientific_source.action_count(config, group)
                records=[[None]*actions for _ in range(scientific_source.prior_particle_count(config))]
                for ai in range(actions):
                    ci=case_cursor;case_cursor += 1;case=ledger.data["cases"][ci];case["status"]="started";ledger.save()
                    action=(observable.action_definition(config,group["family_uid"],ai,group)
                            if settings.get("scientific_profile")
                            else observable.action_definition(config,group["family_uid"],ai))
                    truth=generated["hidden_index"];parameters=generated["parameters"][truth]
                    try:
                        if scientific_source.is_long_cns(config,group["family_uid"]):
                            quality=execute_long_cns_case(config,ledger,group,generated,action,records,ai,ci,identity,local,declared)
                            case.update(status="complete",numerical=quality,raw_reference=f"case_{ci}_reference.npz")
                            ledger.save()
                            continue
                        primary=ledger.execute("physical:"+str(ci),lambda:observable.physical_trajectory(config,group["family_uid"],parameters,action,**({"source_runtime":romani_runtime} if group.get("source_law")=="romani_vm_source" else {})))
                        precise=ledger.execute("physical_precision:"+str(ci),lambda:observable.physical_trajectory(config,group["family_uid"],parameters,action,high_precision=True,**({"source_runtime":romani_runtime} if group.get("source_law")=="romani_vm_source" else {})))
                        middle = primary
                        if group.get("source_law") == "romani_vm_source":
                            middle = ledger.execute("physical_refinement:"+str(ci), lambda:observable.physical_trajectory(
                                config,group["family_uid"],parameters,action,refinement="fine",source_runtime=romani_runtime))
                            precise.update(fine_values=middle["values"], precision_method="independent_fixed_step_refinements")
                        physical_name=f"case_{ci}_physical_producer.json"
                        def serializable(result):
                            return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in result.items()}
                        write_json(local/physical_name, ({"primary":serializable(primary), "reference":serializable(precise),
                            "fine":serializable(middle)} if group.get("source_law")=="romani_vm_source" else
                            {"primary":primary["values"].tolist(),"reference":precise["decimal_values"]}))
                        declared.append(physical_name)
                        for atom in range(scientific_source.prior_particle_count(config)):
                            current=generated["parameters"][atom]
                            def program():
                                estimate=observable.estimated_exposure(config,current,action) if group["family_uid"]==observable.MOLECULAR else None
                                return observable.compiled_program(config,group,generated["support"],atom,current,action,
                                    identity["source_code_sha256"],estimate)
                            records[atom][ai]=ledger.execute(f"program:{ci}:{atom}",program)
                            if group["family_uid"]==observable.MOLECULAR:
                                estimate_reference=ledger.execute(f"chemical_precision:{ci}:{atom}",
                                    lambda:observable.estimated_exposure(config,current,action,high_precision=True))
                                program_arrays=records[atom][ai]["physical_program"]
                                site=next(q["incidence"] for q in observable_profile(config)["quantities"] if q.get("family_uid")==observable.MOLECULAR)
                                ri=observable_profile(config)["region_id"].index(site["anchor_region_id"])
                                co=observable_profile(config)["compartment_id"].index(site["anchor_compartment_id"])
                                expected=estimate_reference["equilibrium_occupancy"]
                                actual=np.asarray(program_arrays["occupancy"])[:,ri,co,0]
                                if not np.allclose(actual,expected,rtol=1e-12,atol=1e-15):
                                    raise ValueError("Estimated chemical program/Decimal equilibrium mismatch")
                        results={}
                        fractions=settings["numerics"]["cns_max_step_interval_fractions"]
                        for label,method,fraction in zip(("coarse","fine","reference"),("RK4","RK4","DOP853"),fractions):
                            physical=precise if label=="reference" else (middle if label=="fine" else primary)
                            law=observable.CoupledObservableLaw(config,group,generated["support"],truth,physical)
                            result=ledger.execute(f"cns:{ci}:{label}",lambda:observable.integrate_coupled(law,method,fraction,ledger.deadline))
                            results[label]=result
                            name=f"case_{ci}_{label}.npz";np.savez_compressed(local/name,**result);declared.append(name)
                        quality=observable.numerical_summary(config,group["family_uid"],parameters,primary,precise,results)
                        case.update(status="complete",numerical=quality,raw_reference=f"case_{ci}_reference.npz")
                    except (ValueError,RuntimeError,FloatingPointError) as error:
                        case.update(status="failed",reason=type(error).__name__+": "+str(error))
                    ledger.save()
                if all(record is not None for row in records for record in row):
                    raw=observable.joint_source(config,group,generated,records)
                    storage = config["native_diffusion_data"].get("group_storage", {})
                    if settings.get("scientific_profile") and storage.get("kind") == "gzip_json_complete_groups":
                        name = f"group_{gi}_conditioning.json.gz"
                        reference = write_joint_source_reference(local/name, raw,
                            max_bytes=storage["compressed_group_max_bytes"][group["family_uid"]])
                        reference["path"] = str(output/name)
                        group["conditioning_reference"] = reference
                    else:
                        name = f"group_{gi}_conditioning.json"
                        write_json(local/name, raw)
                    declared.append(name)
                ledger.save()
                if byte_budget:
                    group_receipt = local/f"group_{gi}_receipt.json"
                    if limit is not None:
                        retained_bytes=stage_source_group(local,output,
                            declared[first_declared:]+[group_receipt.name],byte_budget,staged,retained_bytes,storage)
                    else:
                        byte_budget.record([local/name for name in declared[first_declared:]] + [group_receipt], complete_group=True)
                        retained_bytes=byte_budget.used
                    ledger.data["resource_bytes"] = {"source_files_bytes": retained_bytes,
                        "local_declared_output_bytes":byte_budget.used,
                        "scratch_limit_bytes": byte_budget.limit,
                        "terminal_receipt_reserve_bytes": byte_budget.terminal_reserve}
            prefix_complete=all(case["status"]=="complete" for case in ledger.data["cases"]
                                if case["group_index"] < stop)
            ledger.data["next_group_index"]=stop
            ledger.data["source_status"]=("partial" if prefix_complete and stop<len(ledger.data["groups"])
                else "complete" if prefix_complete else "incomplete")
            if limit is not None:
                ledger.data.setdefault("completed_chunks",[]).append({
                    "first_group_index":start,"stop_group_index":stop,
                    "allocation":os.environ.get("SLURM_JOB_ID"),
                    "source_status":ledger.data["source_status"]})
        except BaseException as error:
            ledger.data.update(source_status="failed",reason=type(error).__name__+": "+str(error))
        finally:
            for case in ledger.data["cases"]:
                if ledger.data["source_status"]!="partial" and case["status"] in {"planned","started"}:
                    case.update(status="missing" if case["status"]=="planned" else "failed",reason="source execution incomplete")
            ledger.data["source_peak_rss_kib"]=max(ledger.data.get("source_peak_rss_kib",0),
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            ledger.save()
            declared.extend(ledger.group_receipt_names)
            if limit is not None:
                conflicts=stage_remaining_source_outputs(local,output,declared,staged)
                if conflicts:
                    ledger.data.update(source_status="failed",unpublished_source_outputs=conflicts)
                    ledger.save()
                    shutil.copyfile(ledger.path,output/"source_receipt.json")
            else:
                for name in dict.fromkeys(declared):
                    if (local/name).is_file():
                        shutil.copyfile(local/name,output/name)
            if limit is not None:
                write_json(chunk_path,{"source_identity":identity,
                    "first_group_index":start,"stop_group_index":stop,
                    "status":"complete" if ledger.data["source_status"] in {"partial","complete"} else "failed",
                    "source_status":ledger.data["source_status"],
                    "source_clock_seconds":ledger.data["source_clock_seconds"],
                    "allocation":os.environ.get("SLURM_JOB_ID"),"host":os.uname().nodename})
                if ledger.data["source_status"]=="complete":
                    shutil.rmtree(local)
    return output



def scientific_group_rows(registry, role=None):
    groups = {}
    for teacher in registry["teachers"]:
        if role is None or teacher["split_name"] == role:
            groups.setdefault(teacher["sample_uid"], []).append(teacher)
    return groups.values()


def scientific_exact_zero_catalogue(receipt):
    """Catalogue conditional proofs without turning them into teacher evidence."""
    catalogue = {(group["family_uid"], group["source_uid"]): {}
                 for group in receipt["groups"]}
    for case in receipt["cases"]:
        group = receipt["groups"][case["group_index"]]
        declarations = catalogue[group["family_uid"], group["source_uid"]]
        for quantity, evidence in case["numerical"]["exact_zero_uncertainty"].items():
            declaration = {"kind": evidence["derivation_kind"],
                           "derivation_ref": evidence["derivation_ref"]}
            if (declaration["kind"] != "identically_zero" or not declaration["derivation_ref"]
                    or evidence["source_snapshot_ref"] != group["source_snapshot_ref"]):
                raise ValueError("Scientific exact-zero proof has an invalid source binding")
            if quantity in declarations and declarations[quantity] != declaration:
                raise ValueError("Scientific exact-zero source declarations conflict: "
                                 + group["family_uid"] + ":" + group["source_uid"] + ":" + quantity)
            declarations[quantity] = declaration
    return catalogue


def archive_failed_scientific_admission(config, receipt, root):
    """Preserve failed derived products before consuming the completed source again."""
    if (not config["native_observable_bridge"].get("scientific_profile")
            or receipt.get("source_status") != "complete"):
        raise ValueError("Admission recovery requires a completed scientific source")
    previous = json.loads((root/"native_consumption.json").read_text())
    if (previous.get("status") not in {"incomplete", "failed"}
            or previous.get("admitted_count") != 0
            or previous.get("source_identity") != receipt["source_identity"]
            or previous.get("declared_cases") != len(receipt["cases"])
            or not (previous.get("error") or previous.get("consumer_error"))):
        raise ValueError("Admission recovery requires its matching failed consumption receipt")
    archive = root/"admission_failures"/str(time.time_ns())
    archive.mkdir(parents=True, exist_ok=False)
    names = [*observable_profile(config)["families"], "native_consumption.json",
             "native_consumption_recheck.json", "simulation_qc_catalog.tsv",
             "native_registry.json", "model_registry.json", "native_config.yaml",
             "train_normalization.json"]
    for name in names:
        path = root/name
        if path.exists():
            path.rename(archive/name)
    return archive


def merge_scientific_registry(combined, registry):
    if not combined:
        combined.update({key: copy.deepcopy(value) for key, value in registry.items()
                         if key not in {"sources", "teachers", "joint_sources", "joint_source_references"}})
        combined.update(sources=[], teachers=[], joint_sources=[], joint_source_references=[])
    if combined["grid"] != registry["grid"]:
        raise ValueError("Scientific group registry changed its family grid")
    existing = {row["source_uid"]: row for row in combined["sources"]}
    for row in registry["sources"]:
        if row["source_uid"] in existing and row != existing[row["source_uid"]]:
            raise ValueError("Scientific group source declaration conflicts")
        if row["source_uid"] not in existing:
            combined["sources"].append(copy.deepcopy(row))
    combined["teachers"].extend(registry["teachers"])
    combined["joint_source_references"].extend(registry["joint_source_references"])


def scientific_index(config, references):
    return {"schema_version": native.FAMILY_INDEX_SCHEMA,
            "quantity_profile_sha256": profile_identity(observable_profile(config)),
            "family_registries": references}


def scientific_group_checks(config, registry, store, *, normalization_context=None):
    roles = {row["split_name"] for row in registry["teachers"]}
    if len(roles) != 1:
        raise ValueError("Complete scientific group crosses source roles")
    role = next(iter(roles))
    selected = copy.deepcopy(config)
    selected["native_diffusion_data"].update(registry_path=str(store/".zattrs"),
                                            source_family_uid=registry["grid"]["family_uid"])
    expected = [row["trajectory_uid"] for row in registry["teachers"]]
    return {kind: consume_dataset(dataset, expected, selected, normalization_context=normalization_context) for kind, dataset in (
        ("direct", MicroenvironmentZarrDataset(store, selected, split_name=role, normalization_context=normalization_context)),
        ("family", MicroenvironmentTeacherFamilyDataset(selected, role, normalization_context=normalization_context)))}


def consume_scientific_groups(config, config_path, receipt, report, report_path, qc, root, *, reuse_completed=False):
    """Admit complete groups, fit training moments once, and retain shared group references."""
    if reuse_completed:
        raise ValueError("Scientific intake requires its immutable completed package, not a producer replay")
    settings = config["native_observable_bridge"]
    storage = config["native_diffusion_data"]["group_storage"]
    exact_catalogue = scientific_exact_zero_catalogue(receipt)
    registries = {family: {} for family in observable_profile(config)["families"]}
    by_group = {}
    for ci, case in enumerate(receipt["cases"]):
        by_group.setdefault(case["group_index"], []).append((ci, case))
    initial_bytes = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
    budget = NativeStorageBudget(storage["retained_max_bytes"], storage["maximum_group_write_bytes"],
                                 storage["maximum_terminal_receipt_bytes"], initial_bytes)
    if (root/"admission_failures").exists():
        budget.record([root/"admission_failures"])
    started = time.monotonic()
    try:
        for gi, group in enumerate(receipt["groups"]):
            budget.reserve_group()
            family = group["family_uid"]
            group_root = root/family/"groups"/str(gi)
            store = group_root/"native_source.zarr"
            if "conditioning_reference" in group:
                reference = group["conditioning_reference"]
                raw, _ = read_joint_source_reference(reference)
            else:
                raw = json.loads((root/f"group_{gi}_conditioning.json").read_text())
                reference = write_joint_source_reference(group_root/"conditioning.json.gz", raw,
                    max_bytes=storage["compressed_group_max_bytes"][family])
            native.parse_joint_source(raw)
            teachers, trajectories, uncertainties = [], [], []
            for ci, case in by_group[gi]:
                program = raw["particle_programs"]["records"][0][0][case["action_index"]]
                with np.load(root/case["raw_reference"]) as data:
                    target = data["trajectory"].astype(np.float32)
                quality = case["numerical"]
                teacher = {**qc[ci], **{key: program[key] for key in
                    ("regimen_uid", "action_uid", "target_program_uid", "program_uid", "posterior_uid")},
                    "content_hash": hashlib.sha256(target.tobytes()).hexdigest(), "trajectory_store_path": str(store),
                    "store_format": "zarr_v2", "trajectory_index": len(teachers),
                    "teacher_family": "full_microenvironment_teacher", "qc_catalog_ref": str(root/"simulation_qc_catalog.tsv"),
                    "source_snapshot_ref": group["source_snapshot_ref"], "uncertainty_origin": "independent_supplied",
                    "exact_zero_uncertainty": quality["exact_zero_uncertainty"],
                    "validity_mask": {"kind": "all_valid", "evidence_ref": str(root/"source_receipt.json")},
                    "calibration_reservation": group["calibration_reservation"]}
                teachers.append(teacher); trajectories.append(target); uncertainties.append(quality["uncertainty"])
            source_row = {"source_uid": group["source_uid"], "public": True, "protected": False,
                "license": (config["native_romani_source"]["source_license"] if group.get("source_law") == "romani_vm_source"
                            else config["controlled_cns_source"]["source_license"]), "release": "controlled_neural_observable_v1",
                **({"external_model_id":config["native_romani_source"]["source_model_id"],
                    "external_source_license_status":"unknown_local_source_readme_has_no_license_statement",
                    "external_source_provenance_ref":config["native_romani_source"]["source_provenance_ref"]}
                   if group.get("source_law") == "romani_vm_source" else {}),
                "snapshot_ref": group["source_snapshot_ref"], "split_evidence_ref": str(root/"source_receipt.json"),
                "unit_evidence_ref": str(config_path), "source_kind": "controlled_simulated",
                "exact_zero_quantities": exact_catalogue[family, group["source_uid"]],
                "predictive_evaluation_units": {"episode_field": "sample_uid", "group_field": "family_uid",
                    "group_kind": "action_family", "independent_episode_units": True, "evidence_ref": str(root/"source_receipt.json")}}
            grid = family_grid(observable_profile(config), family)
            registry = {"schema_version": native.SCHEMA_VERSION, "grid": grid, "sources": [source_row],
                "posterior_family": "finite_joint_particles", "joint_sources": [], "joint_source_references": [reference],
                "teachers": teachers, "programs": [], "observations": []}
            arrays = {"trajectory": np.stack(trajectories), "uncertainty": np.asarray(uncertainties, dtype=np.float32),
                "teacher_confidence": np.asarray([t["teacher_confidence"] for t in teachers], dtype=np.float32),
                "quantity_support": np.stack([quantity_support(grid, teacher["source_uid"]) for teacher in teachers])}
            write_native_microenvironment_store(store, config, registry, arrays)
            budget.record([group_root], complete_group=True)
            merge_scientific_registry(registries[family], registry)
            del raw, registry, arrays, trajectories, uncertainties
        references = {}
        for family, registry in registries.items():
            path = root/family/"native_registry.json"
            write_json(path, registry); references[family] = str(path)
        write_json(root/"native_registry.json", scientific_index(config, references))
        effective = copy.deepcopy(config)
        effective["native_diffusion_data"]["registry_path"] = str(root/"native_registry.json")
        effective_path = root/"native_config.yaml"
        effective_path.write_text(yaml.safe_dump(effective, sort_keys=False))

        admissions = {family: native.NativeRegistryAdmission(registry, effective)
                      for family, registry in registries.items()}

        def training_batches():
            for family, registry in registries.items():
                admission = admissions[family]
                for group in admission.groups("train"):
                    records, arrays, _ = admission.load(group)
                    yield registry["grid"], records, arrays

        normalization = root/"train_normalization.json"
        write_json(normalization, fit_multigrid(training_batches(), effective, streaming=True))
        normalization_context = NativeNormalizationContext(effective)
        model_references = {}
        for family, registry in registries.items():
            combined = {}
            report["prepared_roles"][family] = {}
            admission = admissions[family]
            for group in admission.groups():
                budget.reserve_group()
                records, arrays, _ = admission.load(group)
                teachers = [record["teacher"] for record in records]
                store = Path(teachers[0]["trajectory_store_path"]).parent/"normalized.zarr"
                derived = admission.reindex(records, store, "zarr_v2")
                derived["grid"] = bind_artifact(registry["grid"], normalization, effective, normalization_context=normalization_context)
                write_native_microenvironment_store(store, effective, derived, arrays, normalization_context=normalization_context)
                budget.record([store], complete_group=True)
                checks = scientific_group_checks(effective, derived, store, normalization_context=normalization_context)
                role = teachers[0]["split_name"]
                report["prepared_roles"][family].setdefault(role, []).append(checks)
                merge_scientific_registry(combined, derived)
            path = root/family/"model_registry.json"
            write_json(path, combined); model_references[family] = str(path)
            write_json(report_path, report)
        model_path = root/"model_registry.json"
        write_json(model_path, scientific_index(config, model_references))
        effective["native_diffusion_data"]["registry_path"] = str(model_path)
        effective_path.write_text(yaml.safe_dump(effective, sort_keys=False))
        budget.record([root/"native_registry.json", model_path, effective_path, normalization,
                       *[root/family/name for family in registries for name in ("native_registry.json", "model_registry.json")]])
        normalization_context.verify()
        train_sources = {g["source_uid"] for g in receipt["groups"] if g["role"] == "train"}
        for case in report["cases"]:
            case["native_status"] = "admitted"
        report.update(status="complete", admitted_count=len(report["cases"]),
            source_count=len({g["source_uid"] for g in receipt["groups"]}),
            source_holdout_coverage=any(g["role"] == "validation" and g["source_uid"] not in train_sources for g in receipt["groups"]),
            model_registry_path=str(model_path), normalization_artifact=str(normalization),
            accounted_retained_payload_bytes=budget.used, retained_byte_limit=budget.limit,
            terminal_metadata_reserve_bytes=budget.terminal_reserve)
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      consumer_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(report_path, report)
    return report




def continue_scientific_admission(config, config_path, *, normalization_sha256):
    """Finish derived-group admission without producing or fitting any source."""
    if (not config.get("native_observable_bridge", {}).get("scientific_profile")
            or parameter_schema_from_config(config) is not None):
        raise ValueError("Retained admission continuation applies only to the base scientific source")
    if (not isinstance(normalization_sha256, str) or len(normalization_sha256) != 64
            or any(c not in "0123456789abcdef" for c in normalization_sha256)):
        raise ValueError("Retained admission requires the frozen normalization SHA256")
    effective = copy.deepcopy(config)
    effective["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"].pop(
        "physical_parameter_conditioning", None)
    root = resolve_repo_path(effective["native_observable_bridge"]["output_root"])
    source_path, report_path = root/"source_receipt.json", root/"native_consumption.json"
    receipt = json.loads(source_path.read_text())
    verify_observable_source(receipt, effective)
    previous = json.loads(report_path.read_text())
    if (receipt.get("source_status") != "complete"
            or previous.get("status") not in {"incomplete", "failed"}
            or previous.get("source_identity") != receipt["source_identity"]
            or previous.get("declared_cases") != len(receipt["cases"])
            or any(case.get("status") != "complete" or case.get("numerical", {}).get("reasons")
                   for case in receipt["cases"])):
        raise ValueError("Retained admission requires its matching incomplete receipt and complete valid source")
    effective["native_diffusion_data"]["registry_path"] = str(root/"native_registry.json")
    normalization = root/"train_normalization.json"
    context = NativeNormalizationContext(effective)
    registries = native.load_family_registries(effective, normalization_context=context)
    expected = {case["case_uid"]: (case, receipt["groups"][case["group_index"]]) for case in receipt["cases"]}
    actual = {}
    for family, registry in registries.items():
        if registry["grid"].get("normalization", {}).get("kind") != "physical":
            raise ValueError("Retained admission requires the complete original raw family registry")
        # The digest is caller-supplied frozen authority, not inferred from whatever
        # fitted file happens to remain after an interrupted admission.
        bound = copy.deepcopy(registry["grid"])
        bound["normalization"] = {"kind": "training_scaler", "fit_role": "train",
            "stored_field_coordinates": "source", "artifact_ref": str(normalization), "sha256": normalization_sha256}
        context.spec(bound, effective)
        for row in registry["teachers"]:
            uid = row["trajectory_uid"]
            if uid in actual or uid not in expected:
                raise ValueError("Retained raw registry has duplicate or unexpected cases")
            case, group = expected[uid]
            expected_values = {"source_uid": group["source_uid"], "sample_uid": group["sample_uid"],
                "family_uid": group["group_uid"], "split_name": group["role"],
                "source_holdout_role": group["role"], "qc_grade": "B", "sample_weight": .5,
                "source_snapshot_ref": group["source_snapshot_ref"]}
            if any(row.get(key) != value for key, value in expected_values.items()) or family != group["family_uid"]:
                raise ValueError("Retained raw case source/group/role differs from completed source")
            if Path(row["trajectory_store_path"]) != root/family/"groups"/str(case["group_index"])/"native_source.zarr":
                raise ValueError("Retained raw case store differs from completed source")
            actual[uid] = row
    if set(actual) != set(expected):
        raise ValueError("Retained raw registries are missing complete source cases")
    context.bind_training_registries(registries)
    admissions = {family: native.NativeRegistryAdmission(registry, effective, normalization_context=context)
                  for family, registry in registries.items()}
    storage = effective["native_diffusion_data"]["group_storage"]
    budget = NativeStorageBudget(storage["retained_max_bytes"], storage["maximum_group_write_bytes"],
                                 storage["maximum_terminal_receipt_bytes"])
    # Charge the retained products before adding groups. This runs on a worker,
    # not the management node, and does not rewrite any retained source.
    budget.record([root])
    archive = root/"admission_failures"/str(time.time_ns())
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive/"native_consumption.json", previous)
    identity_paths = ("Libs/utils/data/native_field_normalization.py", "Libs/utils/data/native_diffusion_data.py",
        "Libs/exp/training/pretrain_microenvironment_world_model.py", "Libs/scripts/builders/build_controlled_native_source.py", "Libs/utils/model/microenvironment_teacher.py")
    consumer_files = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in identity_paths}
    report = {"status": "incomplete", "declared_cases": len(receipt["cases"]), "admitted_count": 0,
        "predictive_evaluation": "NOT_RUN", "source_identity": receipt["source_identity"],
        "cases": copy.deepcopy(previous["cases"]), "prepared_roles": {}, "preparation_commands": [],
        "previous_failed_admission_ref": str(archive/"native_consumption.json"),
        "retained_normalization_sha256": normalization_sha256,
        "consumer_commit": subprocess.check_output([*GIT_COMMAND, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "consumer_source_sha256": consumer_files, "source_generation": "NOT_REPLAYED", "normalizer_fit": "NOT_REPLAYED",
        "reused_derived_groups": 0, "written_derived_groups": 0}
    if (len(report["cases"]) != len(expected)
            or {row["case_uid"] for row in report["cases"]} != set(expected)):
        raise ValueError("Previous admission receipt has missing or duplicate source cases")
    started = time.monotonic()
    current_group = None
    try:
        model_references = {}
        for family, registry in registries.items():
            combined = {}
            report["prepared_roles"][family] = {}
            admission = admissions[family]
            for group in admission.groups():
                records, arrays, _ = admission.load(group)
                context.assert_training_records(records)
                teachers = [record["teacher"] for record in records]
                for index, teacher in enumerate(teachers):
                    case, _ = expected[teacher["trajectory_uid"]]
                    if hashlib.sha256(arrays["trajectory"][index].tobytes()).hexdigest() != teacher["content_hash"]:
                        raise ValueError("Retained raw trajectory content differs: "+teacher["trajectory_uid"])
                    declared = np.asarray(case["numerical"]["uncertainty"], dtype=np.float32)
                    if not np.array_equal(arrays["uncertainty"][index], declared):
                        raise ValueError("Retained raw numerical uncertainty differs: "+teacher["trajectory_uid"])
                current_group = teachers[0]["sample_uid"]
                store = Path(teachers[0]["trajectory_store_path"]).parent/"normalized.zarr"
                derived = admission.reindex(records, store, "zarr_v2")
                derived["grid"] = bind_artifact(registry["grid"], normalization, effective,
                                                normalization_context=context)
                if store.exists():
                    descriptor = store/".zattrs"
                    if not descriptor.is_file():
                        raise ValueError("Retained derived group is incomplete: "+str(store))
                    metadata = json.loads(descriptor.read_text())
                    stored = metadata.get("metadata", metadata).get("native_diffusion")
                    if stored != derived:
                        raise ValueError("Retained derived group registry differs: "+str(store))
                    restored, _ = native.load_records(derived, native.join_registry(derived, effective,
                        normalization_context=context), effective, normalization_context=context)
                    if (restored.keys() != arrays.keys() or any(restored[name].dtype != arrays[name].dtype
                            or not np.array_equal(restored[name], arrays[name]) for name in arrays)):
                        raise ValueError("Retained derived group values differ: "+str(store))
                    report["reused_derived_groups"] += 1
                else:
                    budget.reserve_group()
                    write_native_microenvironment_store(store, effective, derived, arrays, normalization_context=context)
                    budget.record([store], complete_group=True)
                    report["written_derived_groups"] += 1
                checks = scientific_group_checks(effective, derived, store, normalization_context=context)
                role = teachers[0]["split_name"]
                report["prepared_roles"][family].setdefault(role, []).append(checks)
                merge_scientific_registry(combined, derived)
            path = root/family/"model_registry.json"
            write_json(path, combined)
            model_references[family] = str(path)
            write_json(report_path, report)
        context.verify()
        model_path = root/"model_registry.json"
        write_json(model_path, scientific_index(effective, model_references))
        effective["native_diffusion_data"]["registry_path"] = str(model_path)
        effective_path = root/"native_config.yaml"
        effective_path.write_text(yaml.safe_dump(effective, sort_keys=False))
        budget.record([model_path, effective_path, archive,
                       *[root/family/"model_registry.json" for family in registries]])
        for case in report["cases"]:
            case["native_status"] = "admitted"
        train_sources = {g["source_uid"] for g in receipt["groups"] if g["role"] == "train"}
        report.update(status="complete", admitted_count=len(expected), source_count=len({g["source_uid"] for g in receipt["groups"]}),
            source_holdout_coverage=any(g["role"] == "validation" and g["source_uid"] not in train_sources for g in receipt["groups"]),
            model_registry_path=str(model_path), normalization_artifact=str(normalization),
            accounted_retained_payload_bytes=budget.used, retained_byte_limit=budget.limit,
            terminal_metadata_reserve_bytes=budget.terminal_reserve)
    except BaseException as error:
        report.update(status="failed", failed_group_uid=current_group, error=type(error).__name__+": "+str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      consumer_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(report_path, report)
    return report


def build_source_scope_normalizations(config, budget, *, normalization_context=None):
    """Fit independent model statistics without changing or duplicating source stores."""
    policy = config.get("manuscript_native_protocol", {}).get("source_comparison")
    if not policy:
        return {}
    owns_context = normalization_context is None
    context = normalization_context or NativeNormalizationContext(config)
    registries = native.load_family_registries(config, normalization_context=context)
    root = resolve_repo_path(policy["normalization_root"])
    root.mkdir(parents=True, exist_ok=True)
    reports = {}
    admissions = {family: native.NativeRegistryAdmission(registry, config, normalization_context=context)
                  for family, registry in registries.items()}
    for scope in ("pure_shared","mixed_shared"):
        scoped = copy.deepcopy(config)
        scoped["native_scientific_training"].update(source_scope=scope, training_subset_percentage=100)
        selected = {row["sample_uid"] for row in scientific_source.selected_group_definitions(scoped,"train")}
        def batches():
            for family, registry in registries.items():
                grid = copy.deepcopy(registry["grid"])
                grid["normalization"] = {"kind":"physical"}
                admission = admissions[family]
                for group in admission.groups("train", sample_uids=selected):
                    records, arrays, _ = admission.load(group)
                    yield grid,records,arrays
        artifact = fit_multigrid(batches(),scoped,streaming=True)
        if {row["sample_uid"] for row in artifact["training_records"]} != selected:
            raise ValueError("Scoped normalizer is missing declared complete training groups")
        grid_ids = {registry["grid"]["normalization"]["sha256"] for registry in registries.values()}
        if len(grid_ids) != 1:
            raise ValueError("Scoped model normalizers require one retained source-grid transform")
        artifact.update(training_data_scope=scientific_source.training_data_scope(scoped),
                        source_grid_normalization_sha256=next(iter(grid_ids)))
        path = root/(scope+".json")
        write_json(path,artifact)
        budget.record([path])
        reports[scope] = {"artifact_ref":str(path), "training_groups":len(selected),
                         "training_trajectories":len(artifact["training_records"]),
                         "source_grid_unchanged":True}
    # A borrowed context belongs to the named admission, which verifies before
    # its complete receipt. Standalone scope fitting must close its own snapshot.
    if owns_context:
        context.verify()
    return reports


def bind_scientific_parameter_groups(config, read_config, source_root, output, receipt, original, report, schema, groups):
    references = {}
    storage = config["native_diffusion_data"]["group_storage"]
    budget = NativeStorageBudget(storage["retained_max_bytes"], storage["maximum_group_write_bytes"],
        storage["maximum_terminal_receipt_bytes"], original["accounted_retained_payload_bytes"])
    # Anonymous source reads and named candidate reads have different scientific
    # configurations. Share each snapshot only with its matching consumers.
    read_context = NativeNormalizationContext(read_config)
    named_context = NativeNormalizationContext(config)
    for family, registry in native.load_family_registries(read_config, normalization_context=read_context).items():
        combined = {}
        report["prepared_roles"][family] = {}
        admission = native.NativeRegistryAdmission(registry, read_config, normalization_context=read_context)
        for selected_group in admission.groups():
            budget.reserve_group()
            records, arrays, _ = admission.load(selected_group)
            teachers = [record["teacher"] for record in records]
            index, group = groups[teachers[0]["sample_uid"]]
            group_root = output/family/"groups"/str(index)
            store = group_root/"native_source.zarr"
            derived = admission.reindex(records, store, "zarr_v2")
            raw = records[0]["joint_source"]
            producer_path = source_root/f"group_{index}_producer.json"
            candidates = {"parameters": json.loads(producer_path.read_text())["parameters"]}
            bound = bind_candidate_parameters(raw, group, candidates, schema, record_ref=str(producer_path),
                                               producer_commit=receipt["source_identity"]["producer_commit"])
            if bound["particle_programs"] != raw["particle_programs"]:
                raise ValueError("Parameter binding changed an original numerical program")
            native.parse_joint_source(bound)
            reference = write_joint_source_reference(group_root/"conditioning.json.gz", bound,
                max_bytes=storage["compressed_group_max_bytes"][family])
            derived.update(joint_sources=[], joint_source_references=[reference])
            write_native_microenvironment_store(store, config, derived, arrays, normalization_context=named_context)
            budget.record([group_root], complete_group=True)
            loaded, _ = native.load_records(derived, native.join_registry(derived, config, normalization_context=named_context),
                                            config, normalization_context=named_context)
            if any(not np.array_equal(loaded[key], arrays[key]) for key in arrays):
                raise ValueError("Derived group changed retained numeric arrays")
            checks = scientific_group_checks(config, derived, store, normalization_context=named_context)
            report["prepared_roles"][family].setdefault(teachers[0]["split_name"], []).append(checks)
            report["groups"].append({"group_uid": group["group_uid"], "particle_count": len(group["particle_uids"]),
                "parameter_names": [row["name"] for row in parameter_declarations(schema, group["source_uid"], family)[1]], "source_record_ref": str(producer_path)})
            merge_scientific_registry(combined, derived)
        path = output/family/"model_registry.json"
        write_json(path, combined); references[family] = str(path)
    write_json(native.registry_path(config), scientific_index(config, references))
    budget.record([native.registry_path(config), *[Path(path) for path in references.values()]])
    scope_normalizers = build_source_scope_normalizations(config, budget, normalization_context=named_context)
    read_context.verify()
    named_context.verify()
    for case in report["cases"]:
        case["native_status"] = "admitted"
    report.update(status="complete", source_scope_normalizers=scope_normalizers,
                  admitted_count=len(report["cases"]), array_values_unchanged=True,
                  compiled_programs_unchanged=True, model_registry_path=str(native.registry_path(config)),
                  normalization_artifact=original["normalization_artifact"],
                  accounted_retained_payload_bytes=budget.used, retained_byte_limit=budget.limit,
                  terminal_metadata_reserve_bytes=budget.terminal_reserve)


def consume_observable_parameter_conditioning(config, config_path):
    """Derive complete candidate metadata/stores without changing the spent source."""
    settings = config["native_observable_bridge"]
    source_root = resolve_repo_path(settings["output_root"])
    receipt = json.loads((source_root / "source_receipt.json").read_text())
    verify_observable_source(receipt, config)
    original = json.loads((source_root / "native_consumption.json").read_text())
    if (receipt["source_status"] != "complete" or original["status"] != "complete"
            or original["admitted_count"] != settings["budget"]["cases"]):
        raise ValueError("Retained parameter binding requires the complete original source admission")
    schema = parameter_schema_from_config(config)
    if schema is None:
        raise ValueError("Retained parameter binding requires the named-candidate policy")
    output = native.registry_path(config).parent
    if output == source_root:
        raise ValueError("Derived parameter conditioning cannot overwrite original source artifacts")
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "native_consumption.json"
    report = {"status": "started", "operation": "bind_retained_physical_candidates",
              "declared_cases": settings["budget"]["cases"], "admitted_count": 0,
              "source_identity": receipt["source_identity"], "consumer_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "original_consumption_ref": str(source_root / "native_consumption.json"),
              "original_runtime_ref": str(source_root / "native_consumer_check.json"),
              "source_operations": 0, "model_initializations": 0,
              "full_native_complete_prior_check": "NOT_RUN", "predictive_evaluation": "NOT_RUN",
              "groups": [], "prepared_roles": {}, "cases": copy.deepcopy(original["cases"])}
    for case in report["cases"]:
        case["native_status"] = "candidate"
    write_json(report_path, report)
    started = time.monotonic()
    # Original metadata remains readable at its recorded incomplete-prior scope.
    # It is never emitted as the repaired model input.
    read_config = copy.deepcopy(config)
    read_config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"].pop(
        "physical_parameter_conditioning")
    read_config["native_diffusion_data"]["registry_path"] = str(source_root / "model_registry.json")
    groups = {group["sample_uid"]: (index, group) for index, group in enumerate(receipt["groups"])}
    references = {}
    try:
        if settings.get("scientific_profile"):
            bind_scientific_parameter_groups(config, read_config, source_root, output, receipt, original, report, schema, groups)
            return report
        for family, original_registry in native.load_family_registries(read_config).items():
            records = native.join_registry(original_registry, read_config)
            arrays, _ = native.load_records(original_registry, records, read_config)
            family_root = output / family
            store = family_root / "native_source.zarr"
            derived = native.reindex_registry(original_registry, records, store, "zarr_v2")
            augmented = []
            for raw in derived["joint_sources"]:
                if len(raw["support"]["sample_uids"]) != 1:
                    raise ValueError("Retained observable groups require one complete sample")
                index, group = groups[raw["support"]["sample_uids"][0]]
                path = source_root / f"group_{index}_producer.json"
                candidates = {"parameters": json.loads(path.read_text())["parameters"]}
                bound = bind_candidate_parameters(raw, group, candidates, schema,
                    record_ref=str(path), producer_commit=receipt["source_identity"]["producer_commit"])
                if bound["particle_programs"] != raw["particle_programs"]:
                    raise ValueError("Parameter binding changed an original numerical program")
                native.parse_joint_source(bound)
                augmented.append(bound)
                report["groups"].append({"group_uid": group["group_uid"], "particle_count": len(group["particle_uids"]),
                    "parameter_names": [row["name"] for row in parameter_declarations(schema, raw["support"]["source_uid"], family)[1]],
                    "source_record_ref": str(path)})
            derived["joint_sources"] = augmented
            write_native_microenvironment_store(store, config, derived, arrays)
            loaded, _ = native.load_records(derived, native.join_registry(derived, config), config)
            if any(not np.array_equal(loaded[key], arrays[key]) for key in arrays):
                raise ValueError("Derived store changed retained numeric arrays")
            path = family_root / "model_registry.json"
            write_json(path, derived)
            references[family] = str(path)
            selected = copy.deepcopy(config)
            selected["native_diffusion_data"].update(registry_path=str(path), source_family_uid=family)
            report["prepared_roles"][family] = {}
            for role in observable_source_roles(config):
                expected = [row["trajectory_uid"] for row in derived["teachers"] if row["split_name"] == role]
                checks = {}
                for kind in ("direct", "family"):
                    dataset = (MicroenvironmentZarrDataset(store, selected, split_name=role) if kind == "direct"
                               else MicroenvironmentTeacherFamilyDataset(selected, role))
                    checks[kind] = consume_dataset(dataset, expected, selected)
                report["prepared_roles"][family][role] = checks
            del arrays, loaded, records, original_registry, derived
            gc.collect()
        index = {"schema_version": native.FAMILY_INDEX_SCHEMA,
                 "quantity_profile_sha256": profile_identity(observable_profile(config)),
                 "family_registries": references}
        write_json(native.registry_path(config), index)
        for case in report["cases"]:
            case["native_status"] = "admitted"
        report.update(status="complete", admitted_count=len(report["cases"]),
                      array_values_unchanged=True, compiled_programs_unchanged=True,
                      model_registry_path=str(native.registry_path(config)),
                      normalization_artifact=original["normalization_artifact"])
    except BaseException as error:
        for case in report["cases"]:
            case["native_status"] = "consumer_failed"
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["consumer_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(report_path, report)
    return report


def consume_observable_source(config, config_path, *, reuse_completed=False, recover_admission=False):
    if reuse_completed and config["native_observable_bridge"].get("scientific_profile"):
        raise ValueError("Scientific intake requires its immutable completed package, not a producer replay")
    if config["native_observable_bridge"].get("scientific_profile"):
        config = copy.deepcopy(config)
        config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"].pop(
            "physical_parameter_conditioning", None)
    settings=config["native_observable_bridge"];root=resolve_repo_path(settings["output_root"])
    receipt=json.loads((root/"source_receipt.json").read_text())
    verify_observable_source(receipt,config)
    archive = archive_failed_scientific_admission(config, receipt, root) if recover_admission else None
    report_path=root/("native_consumption_recheck.json" if reuse_completed else "native_consumption.json")
    if reuse_completed and report_path.exists():
        raise FileExistsError("Observable consumption recheck receipt already exists")
    report={"status":"incomplete","declared_cases":settings["budget"]["cases"],"admitted_count":0,
        "predictive_evaluation":"NOT_RUN","source_identity":receipt["source_identity"],"cases":[],
        "prepared_roles":{},"preparation_commands":[]}
    if archive is not None:
        report["previous_failed_admission_ref"] = str(archive/"native_consumption.json")
    qc=[]
    for case in receipt["cases"]:
        group=receipt["groups"][case["group_index"]]
        reasons=list(case.get("numerical",{}).get("reasons",[]))
        if case["status"]!="complete":reasons.append(case.get("reason",case["status"]))
        row={"trajectory_uid":case["case_uid"],"source_uid":group.get("source_uid",settings["source_uid"]),"sample_uid":group["sample_uid"],
            "family_uid":group["group_uid"],"qc_grade":"B" if not reasons else "D",
            "qc_use":"main_pretraining" if not reasons else "reject","sample_weight":.5 if not reasons else 0.,
            "split_name":group["role"],"source_holdout_role":group["role"],
            "teacher_confidence":case.get("numerical",{}).get("teacher_confidence",0.)}
        if settings.get("scientific_profile"):
            row.update({key:group[key] for key in ("partition_uid","ood_class","source_law","condition","training_subset_percentages")
                        if key in group})
        qc.append(row)
        report["cases"].append({"case_uid":case["case_uid"],"role":group["role"],"reasons":reasons,
                               "source_status":case["status"],"native_status":"candidate" if not reasons else "rejected"})
    write_tsv(root/"simulation_qc_catalog.tsv",list(dict.fromkeys(key for row in qc for key in row)),qc)
    write_json(report_path,report)
    if receipt["source_status"]!="complete" or any(row["reasons"] for row in report["cases"]):
        return report
    if settings.get("scientific_profile"):
        return consume_scientific_groups(config, config_path, receipt, report, report_path, qc, root,
                                         reuse_completed=reuse_completed)
    references={}
    for family in observable_profile(config)["families"]:
        family_root=root/family;family_root.mkdir(exist_ok=reuse_completed);store=family_root/"native_source.zarr"
        selected=[(ci,c) for ci,c in enumerate(receipt["cases"]) if receipt["groups"][c["group_index"]]["family_uid"]==family]
        sources=[];teachers=[];trajectories=[];uncertainties=[];exact={};exact_by_source={}
        for ci,case in selected:
            group=receipt["groups"][case["group_index"]]
            raw=json.loads((root/f'group_{case["group_index"]}_conditioning.json').read_text())
            native.parse_joint_source(raw)
            if raw["source_posterior_uid"] not in {value["source_posterior_uid"] for value in sources}:sources.append(raw)
            program=raw["particle_programs"]["records"][0][0][case["action_index"]]
            with np.load(root/case["raw_reference"]) as data:target=data["trajectory"].astype(np.float32)
            quality=case["numerical"]
            teacher={**qc[ci],**{key:program[key] for key in ("regimen_uid","action_uid","target_program_uid","program_uid","posterior_uid")},
                "content_hash":hashlib.sha256(target.tobytes()).hexdigest(),"trajectory_store_path":str(store),
                "store_format":"zarr_v2","trajectory_index":len(teachers),"teacher_family":"full_microenvironment_teacher",
                "qc_catalog_ref":str(root/"simulation_qc_catalog.tsv"),"source_snapshot_ref":group.get("source_snapshot_ref",settings["snapshot_ref"]),
                "uncertainty_origin":"independent_supplied","exact_zero_uncertainty":quality["exact_zero_uncertainty"],
                "validity_mask":{"kind":"all_valid","evidence_ref":str(root/"source_receipt.json")},
                "calibration_reservation":group["calibration_reservation"]}
            source_exact = exact_by_source.setdefault(group.get("source_uid",settings["source_uid"]),{})
            for name,evidence in quality["exact_zero_uncertainty"].items():
                exact[name]={"kind":"identically_zero","derivation_ref":evidence["derivation_ref"]}
                source_exact[name] = dict(exact[name])
            teachers.append(teacher);trajectories.append(target);uncertainties.append(quality["uncertainty"])
        source_row={"source_uid":settings["source_uid"],"public":True,"protected":False,
            "license":config["controlled_cns_source"]["source_license"],"release":"controlled_neural_observable_v1",
            "snapshot_ref":settings["snapshot_ref"],"split_evidence_ref":str(root/"source_receipt.json"),
            "unit_evidence_ref":str(config_path),"source_kind":"controlled_simulated","exact_zero_quantities":exact,
            "predictive_evaluation_units":{"episode_field":"sample_uid","group_field":"family_uid",
                "group_kind":"action_family","independent_episode_units":True,"evidence_ref":str(root/"source_receipt.json")}}
        source_rows = []
        seen_sources = set()
        for _,case in selected:
            group = receipt["groups"][case["group_index"]]
            source_uid = group.get("source_uid",settings["source_uid"])
            snapshot = group.get("source_snapshot_ref",settings["snapshot_ref"])
            if (source_uid,snapshot) not in seen_sources:
                source_rows.append(dict(source_row,source_uid=source_uid,snapshot_ref=snapshot,exact_zero_quantities=exact_by_source.get(source_uid,{})))
                seen_sources.add((source_uid,snapshot))
        grid=family_grid(observable_profile(config),family)
        registry={"schema_version":native.SCHEMA_VERSION,"grid":grid,"sources":source_rows,"posterior_family":"finite_joint_particles",
                  "joint_sources":sources,"teachers":teachers,"programs":[],"observations":[]}
        arrays={"trajectory":np.stack(trajectories),"uncertainty":np.asarray(uncertainties,dtype=np.float32),
                "teacher_confidence":np.asarray([t["teacher_confidence"] for t in teachers],dtype=np.float32),
                "quantity_support":np.stack([quantity_support(grid, teacher["source_uid"]) for teacher in teachers])}
        if store.exists():
            if not reuse_completed:
                raise FileExistsError("Observable source store already exists")
            stored=json.loads((store/".zattrs").read_text())["native_diffusion"]
            if stored!=registry:
                raise ValueError("Existing observable store/source registry differs")
        else:
            write_native_microenvironment_store(store,config,registry,arrays)
        path=family_root/"native_registry.json"
        if path.exists() and json.loads(path.read_text())!=registry:
            raise ValueError("Existing observable registry differs from retained source")
        if not path.exists():write_json(path,registry)
        references[family]=str(path)
    index={"schema_version":native.FAMILY_INDEX_SCHEMA,"quantity_profile_sha256":profile_identity(observable_profile(config)),
           "family_registries":references}
    if (root/"native_registry.json").exists() and json.loads((root/"native_registry.json").read_text())!=index:
        raise ValueError("Existing family index differs from retained source")
    if not (root/"native_registry.json").exists():write_json(root/"native_registry.json",index)
    effective=copy.deepcopy(config);effective["native_diffusion_data"]["registry_path"]=str(root/"native_registry.json")
    effective_path=root/"native_config.yaml";effective_path.write_text(yaml.safe_dump(effective,sort_keys=False))
    normalization=root/"train_normalization.json"
    if not normalization.exists():
        report["preparation_commands"].append(call_preparation(["--config",str(effective_path),"--split","train",
            "--max-rows",str(settings["budget"]["cases"]),"--fit-field-normalization","--output",str(normalization)]))
    for family in references:
        report["prepared_roles"][family]={}
        for role in observable_source_roles(config):
            store=root/family/(role+"_normalized.zarr")
            if not store.exists():
                report["preparation_commands"].append(call_preparation(["--config",str(effective_path),"--family",family,"--split",role,
                    "--max-rows",str(settings["budget"]["cases"]),"--normalization-artifact",str(normalization),"--output",str(store)]))
            selected=copy.deepcopy(effective);selected["native_diffusion_data"].update(registry_path=str(store/".zattrs"),source_family_uid=family)
            expected=[c["case_uid"] for c in receipt["cases"] if receipt["groups"][c["group_index"]]["family_uid"]==family
                      and receipt["groups"][c["group_index"]]["role"]==role]
            checks={}
            for kind in ("direct","family"):
                dataset=MicroenvironmentZarrDataset(store,selected,split_name=role) if kind=="direct" else MicroenvironmentTeacherFamilyDataset(selected,role)
                checks[kind]=consume_dataset(dataset,expected,selected)
            report["prepared_roles"][family][role]=checks
    model_references={}
    for family in references:
        role_registries=[json.loads((root/family/(role+"_normalized.zarr")/".zattrs").read_text())["native_diffusion"]
                         for role in observable_source_roles(config)]
        combined=copy.deepcopy(role_registries[0])
        if any(row["grid"]!=combined["grid"] for row in role_registries):
            raise ValueError("All source roles must share the same training-fitted coordinate transform")
        for collection,identity_key in (("teachers","trajectory_uid"),("joint_sources","source_posterior_uid"),
                                        ("sources","source_uid"),("programs","program_uid"),("observations","posterior_uid")):
            unique={}
            for registry in role_registries:
                for row in registry.get(collection,[]):
                    key=row[identity_key]
                    if key in unique and unique[key]!=row:
                        raise ValueError("Conflicting normalized source role identity: "+key)
                    unique[key]=row
            combined[collection]=list(unique.values())
        native.join_registry(combined,config)
        path=root/family/"model_registry.json"
        if path.exists() and json.loads(path.read_text())!=combined:
            raise ValueError("Existing model registry differs from retained role stores")
        if not path.exists():write_json(path,combined)
        model_references[family]=str(path)
    model_index={"schema_version":native.FAMILY_INDEX_SCHEMA,"quantity_profile_sha256":profile_identity(observable_profile(config)),
        "family_registries":model_references}
    model_path=root/"model_registry.json"
    if model_path.exists() and json.loads(model_path.read_text())!=model_index:
        raise ValueError("Model family registry differs from the fitted training source")
    if not model_path.exists():write_json(model_path,model_index)
    effective["native_diffusion_data"]["registry_path"]=str(model_path)
    effective_path.write_text(yaml.safe_dump(effective,sort_keys=False))
    for row in report["cases"]:row["native_status"]="admitted"
    report.update(status="complete",admitted_count=len(report["cases"]),source_count=1,source_holdout_coverage=False,
        model_registry_path=str(model_path),normalization_artifact=str(normalization),
        consumer_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(report_path,report)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="Libs/configs/exp/controlled_native_source.yaml")
    parser.add_argument("--consume-only",action="store_true")
    parser.add_argument("--continue-source",action="store_true",
                        help="Continue a verified complete-group source prefix without replay")
    parser.add_argument("--recover-admission", action="store_true",
                        help="Preserve failed admission products and consume a completed scientific source")
    parser.add_argument("--continue-admission", action="store_true",
                        help="Admit missing derived groups while retaining the complete raw source and frozen train normalizer")
    parser.add_argument("--normalization-sha256", help="Frozen fitted artifact digest required by retained admission")
    args=parser.parse_args()
    if sum((args.consume_only, args.continue_source, args.recover_admission, args.continue_admission)) > 1:
        parser.error("Consumption and source continuation are distinct operations")
    config_path=resolve_repo_path(args.config);config=load_config(str(config_path))
    if args.normalization_sha256 and not args.continue_admission:
        parser.error("--normalization-sha256 requires --continue-admission")
    if args.continue_admission:
        report = continue_scientific_admission(config, config_path, normalization_sha256=args.normalization_sha256)
        print(json.dumps(report, allow_nan=False), flush=True)
        return 0 if report["status"] == "complete" else 2
    if args.recover_admission:
        if (not config.get("native_observable_bridge", {}).get("scientific_profile")
                or parameter_schema_from_config(config) is not None):
            raise ValueError("Admission recovery applies only to the base scientific source")
        report = consume_observable_source(config, config_path, recover_admission=True)
        print(json.dumps(report, allow_nan=False), flush=True)
        return 0 if report["status"] == "complete" else 2
    if config.get("native_observable_bridge",{}).get("enabled"):
        if parameter_schema_from_config(config) is not None:
            if not args.consume_only:
                raise ValueError("Named physical parameter conditioning consumes retained inputs only")
            report = consume_observable_parameter_conditioning(config, config_path)
            print(json.dumps(report, allow_nan=False), flush=True)
            return 0 if report["status"] == "complete" else 2
        if not args.consume_only:
            root=execute_observable_source(config,config_path,continue_source=args.continue_source)
            progress=json.loads((root/"source_receipt.json").read_text())
            if progress["source_status"]=="partial":
                print(json.dumps({"status":"source_chunk_complete","source_status":"partial",
                    "next_group_index":progress["next_group_index"],
                    "declared_groups":len(progress["groups"]),"source_admission":"NOT_RUN",
                    "continuation_required":True}),flush=True)
                return 0
        try:
            report=consume_observable_source(config,config_path,reuse_completed=args.consume_only)
        except Exception as error:
            root=resolve_repo_path(config["native_observable_bridge"]["output_root"])
            path=root/("native_consumption_recheck.json" if args.consume_only else "native_consumption.json")
            report=json.loads(path.read_text()) if path.exists() else {"predictive_evaluation":"NOT_RUN","admitted_count":0}
            report.update(status="incomplete",consumer_error=type(error).__name__+": "+str(error))
            write_json(path,report)
        print(json.dumps(report,allow_nan=False),flush=True)
        return 0 if report["status"]=="complete" else 2
    if args.continue_source:
        raise ValueError("Legacy source execution cannot be continued")
    source=controlled.definition(config);root=resolve_repo_path(source["output_root"])
    if not args.consume_only:execute_source(config,config_path)
    if args.consume_only and (root/"native_consumption_recheck.json").exists():
        raise FileExistsError("Consumption recheck receipt already exists")
    try:report=consume_source(config,config_path,reuse_completed=args.consume_only)
    except Exception as error:
        previous=root/("native_consumption_recheck.json" if args.consume_only else "native_consumption.json")
        report=json.loads(previous.read_text()) if previous.exists() else {}
        if "cases" not in report:
            original=json.loads((root/"source_receipt.json").read_text())
            report["cases"]=[{"case_uid":row["case_uid"],"source_status":row["status"],"native_status":"consumer_failed"}
                             for row in original["cases"]]
        for row in report["cases"]:
            if row["native_status"]=="candidate":row["native_status"]="consumer_failed"
        report.update(status="incomplete",declared_cases=16,
            admitted_count=sum(row["native_status"]=="admitted" for row in report["cases"]),
            predictive_evaluation="NOT_RUN",consumer_error=type(error).__name__+": "+str(error))
        write_json(previous,report)
    print(json.dumps(report,allow_nan=False),flush=True)
    return 0 if report["status"]=="complete" else 2


if __name__=="__main__":raise SystemExit(main())
