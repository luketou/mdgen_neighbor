"""Compute the #43 provenance audit numbers from pinned upstream clones and RCSB."""
import csv, hashlib, json, subprocess, sys, urllib.request
from pathlib import Path

mdgen, alphaflow, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SPLITS = ["atlas", "atlas_train", "atlas_val", "atlas_test"]


def head(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(repo, name):
    return {r["name"]: r for r in csv.DictReader(open(repo / "splits" / f"{name}.csv", newline=""))}


def rcsb(pdb):
    e = json.load(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}"))
    info = e["rcsb_accession_info"]
    return {"deposit_date": info["deposit_date"][:10], "initial_release_date": info["initial_release_date"][:10]}


def find_line(path, needle):
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        if needle in line:
            return {"line": i, "text": line.rstrip()}
    raise SystemExit(f"{needle!r} not found in {path}")


s = {n: load(mdgen, n) for n in SPLITS}
assigned = set(s["atlas_train"]) | set(s["atlas_val"]) | set(s["atlas_test"])
missing = sorted(set(s["atlas"]) - assigned)


def split_stats(n):
    rows = s[n].values()
    lens = [len(r["seqres"]) for r in rows]
    st = {"count": len(lens), "min_length": min(lens), "max_length": max(lens)}
    if n != "atlas":
        dates = sorted(r["release_date"] for r in rows)
        st.update(min_release_date=dates[0], max_release_date=dates[-1])
    return st


train_over_1024 = sorted(len(r["seqres"]) for r in s["atlas_train"].values() if len(r["seqres"]) > 1024)
test = s["atlas_test"]
longest_test = max(test, key=lambda k: len(test[k]["seqres"]))
spot = ["6sms_A", "6lrd_A", "6l3r_E", "6o2v_A", "1k5n_A"]
rows_all = {**s["atlas_train"], **s["atlas_val"], **test}

frames_10ps = 100_000 // 10 + 1
audit = {
    "generated_by": "python -I make_audit.py <mdgen clone> <alphaflow clone> audit.json (run 2026-10-24)",
    "inputs": {
        "mdgen": {"repo": "https://github.com/bjing2016/mdgen", "commit": head(mdgen)},
        "alphaflow": {
            "repo": "https://github.com/bjing2016/alphaflow",
            "commit": head(alphaflow),
            "splits_last_changed": subprocess.check_output(
                ["git", "-C", str(alphaflow), "log", "-1", "--format=%H %ad %s", "--date=short", "--", "splits/atlas.csv", "splits/atlas_test.csv"],
                text=True,
            ).strip(),
        },
        "rcsb_api": "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
        "papers": {
            "mdgen": "https://arxiv.org/abs/2409.17808",
            "alphaflow": "https://arxiv.org/abs/2402.04845",
            "read_on": "2026-10-23",
        },
    },
    "split_files_sha256": {
        n: {"mdgen": sha(mdgen / "splits" / f"{n}.csv"), "alphaflow": sha(alphaflow / "splits" / f"{n}.csv")}
        for n in SPLITS
    },
    "split_stats": {n: split_stats(n) for n in SPLITS},
    "assigned_total": len(assigned),
    "unassigned": {
        m: {"length": len(s["atlas"][m]["seqres"]), **rcsb(m[:4])} for m in missing
    },
    "test_period_total": len(s["atlas_test"]) + sum(
        rcsb(m[:4])["initial_release_date"] >= "2019-05-01" for m in missing
    ),
    "6sup_A_line_in_atlas_csv": find_line(mdgen / "splits" / "atlas.csv", "6sup_A,")["line"],
    "train_lengths_over_1024": train_over_1024,
    "test_seqlen_matches_seqres": all(int(r["seqlen"]) == len(r["seqres"]) for r in test.values()),
    "longest_test": {"name": longest_test, "length": len(test[longest_test]["seqres"])},
    "release_date_spot_check": {
        n: {"csv_release_date": rows_all[n]["release_date"], "split": next(k for k in SPLITS[1:] if n in s[k]), **rcsb(n[:4])}
        for n in spot
    },
    "frame_arithmetic": {
        "atlas_interval_ps": 10,
        "replica_length_ps": 100_000,
        "frames_at_10ps": frames_10ps,
        "upstream_stride": 40,
        "frames_after_stride": len(range(0, frames_10ps, 40)),
        "training_interval_ps": 10 * 40,
        "num_frames": 250,
        "possible_frame_starts": len(range(len(range(0, frames_10ps, 40)) - 250)),
        "training_frames": [0, 249],
        "training_span_ns": [0.0, 249 * 400 / 1000],
        "mirror_interval_ps": 100,
        "mirror_frames": 100_000 // 100 + 1,
        "mirror_stride": 4,
        "mirror_frames_after_stride": len(range(0, 100_000 // 100 + 1, 4)),
    },
    "6sup_A_over_longest_test": round(2128 / len(test[longest_test]["seqres"]), 2),
    "source_lines": {
        "mdgen/README.md": {
            "atlas_download": find_line(mdgen / "README.md", "download_atlas.sh"),
            "atlas_prep_comment": find_line(mdgen / "README.md", "40 * 10 ps = 400 ps"),
            "atlas_prep_command": find_line(mdgen / "README.md", "--suffix _i40 --stride 40"),
            "atlas_train_command": find_line(mdgen / "README.md", "--num_frames 250 --batch_size 1"),
            "weights_statement": find_line(mdgen / "README.md", "model weights used in the paper"),
        },
        "mdgen/mdgen/dataset.py": {
            "frame_start": find_line(mdgen / "mdgen/dataset.py", "np.random.choice(np.arange(arr.shape[0]"),
            "crop_branch": find_line(mdgen / "mdgen/dataset.py", "if L > self.args.crop"),
            "pad_branch": find_line(mdgen / "mdgen/dataset.py", "elif L < self.args.crop"),
        },
        "mdgen/mdgen/parsing.py": {"abs_pos_emb_flag": find_line(mdgen / "mdgen/parsing.py", "'--abs_pos_emb'")},
        "mdgen/mdgen/model/latent_model.py": {
            "pos_embed_guard": find_line(mdgen / "mdgen/model/latent_model.py", "if args.abs_pos_emb:"),
            "pos_embed_add": find_line(mdgen / "mdgen/model/latent_model.py", "x = x + self.pos_embed"),
        },
        "mdgen/sim_inference.py": {"full_length": find_line(mdgen / "sim_inference.py", "L = len(seqres)")},
        "alphaflow/scripts/download_atlas.sh": {"zip": find_line(alphaflow / "scripts/download_atlas.sh", "_protein.zip")},
    },
    "paper_quotes": {
        "mdgen_4.4": "we generate samples with 250 frames and 400 ps timestep, such that a single sample emulates the 100 ns ATLAS reference trajectory",
        "mdgen_B.3_split": "We split the dataset into 1265 training, 39 validation, and 82 test proteins by PDB release date following Jing et al. [2024]",
        "mdgen_B.3_stride": "subsample every 40 frames, obtaining a training target with 250 frames",
        "mdgen_table4_caption": "Median results on test protein ensembles (n=82)",
        "mdgen_B.3_crop": "We train with random crops of up to 256 residues, but draw samples for the full protein at inference time",
        "alphaflow_4.1": "Using training and validation cutoffs of May 1, 2018 and May 1, 2019, we obtain train/val/test splits of 1265/39/82 ensembles (2 excluded due to length)",
        "alphaflow_B.2_frames": "3 replicate simulations of length 100 ns are provided, each with 10k frames",
        "alphaflow_B.2_test": "all 84 targets whose corresponding PDB entries were deposited after May 1, 2019, minus the two targets with sequence length greater than 1024",
    },
}
out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
