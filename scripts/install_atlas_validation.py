"""Install only the three audited ATLAS candidates; run on a CPU compute node."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import subprocess
import time
import zipfile


CANDIDATES = {"6sms_A": 724, "6lrd_A": 705, "6l3r_E": 669}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(name, expected_residues, destination, run):
    url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{name}/{name}_protein.zip"
    final = destination / name
    if final.exists():
        raise RuntimeError(f"Refusing to overwrite existing destination: {final}")
    if shutil.disk_usage(destination).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB filesystem space available")
    archive = run / "cache" / f"{name}_protein.zip.part"
    stage = destination / f".{name}-{os.environ['SLURM_JOB_ID']}.partial"
    stage.mkdir()  # A unique staging directory; never merge with existing data.
    print(f"Downloading {name}: {url}", flush=True)
    subprocess.run([
        "curl", "--fail", "--location", "--show-error", "--silent",
        "--connect-timeout", "30", "--max-time", "1800",
        "--speed-limit", "1024", "--speed-time", "120",
        "--retry", "2", "--retry-delay", "15", "--retry-max-time", "5400",
        "--output", str(archive), url,
    ], check=True)
    archive_hash = sha256(archive)
    print(f"Downloaded {name}: {archive.stat().st_size} bytes; verifying ZIP", flush=True)
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if sum(info.file_size for info in entries) > 20 * 1024**3:
            raise RuntimeError("Archive expands beyond the 20 GiB safety limit")
        paths = set()
        for info in entries:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise RuntimeError(f"Unsafe ZIP path: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError(f"Unexpected ZIP symlink: {info.filename}")
            if str(path) in paths:
                raise RuntimeError(f"Duplicate ZIP member: {info.filename}")
            paths.add(str(path))
        corrupt = source.testzip()
        if corrupt:
            raise RuntimeError(f"ZIP CRC mismatch: {corrupt}")
        source.extractall(stage)

    pdb_name = f"{name}.pdb"
    matches = list(stage.rglob(pdb_name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pdb_name}, found {len(matches)}")
    payload = matches[0].parent
    required = [pdb_name] + [f"{name}_prod_R{i}_fit.xtc" for i in (1, 2, 3)]
    for filename in required:
        path = payload / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty required file: {filename}")
        if path.suffix == ".xtc":
            with path.open("rb") as stream:
                header = stream.read(8)
            if len(header) != 8 or struct.unpack(">ii", header)[0] != 1995:
                raise RuntimeError(f"Invalid XTC header: {filename}")

    residues = set()
    for line in (payload / pdb_name).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            residues.add((line[21:22], line[22:27]))
    files = [{"path": str(path.relative_to(payload)), "bytes": path.stat().st_size,
              "sha256": sha256(path)} for path in sorted(payload.rglob("*")) if path.is_file()]
    record = {
        "name": name, "status": "complete", "source_url": url,
        "archive_bytes": archive.stat().st_size, "archive_sha256": archive_hash,
        "split": "local atlas_test.csv", "seqres_length": expected_residues,
        "pdb_ca_residues": len(residues), "files": files,
        "validation": "ZIP CRC, required files, XTC header, SHA256; no full MD parsing",
        "training_independence": "not yet verified", "destination": str(final),
    }
    (payload / "installation.json").write_text(json.dumps(record, indent=2) + "\n")
    os.rename(payload, final)
    if stage.exists():
        shutil.rmtree(stage)
    archive.unlink()  # Remove only this run's successfully verified ZIP temporary file.
    print(f"Installed {name}: {final}; PDB CA residues={len(residues)}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Installation must run inside a Slurm CPU job")
    args.destination.mkdir(parents=True, exist_ok=True)
    report = {"job_id": os.environ["SLURM_JOB_ID"], "started_unix": time.time(),
              "destination": str(args.destination), "proteins": [], "status": "running"}
    report_path = args.run_dir / "results" / "installation_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for name, length in CANDIDATES.items():
        try:
            item = install(name, length, args.destination, args.run_dir)
        except Exception as error:
            item = {"name": name, "status": "failed", "error": str(error)}
            print(f"FAILED {name}: {error}", flush=True)
        report["proteins"].append(item)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete" if all(x["status"] == "complete" for x in report["proteins"]) else "failed"
    report["finished_unix"] = time.time()
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Final status: {report['status']}; report={report_path}", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
