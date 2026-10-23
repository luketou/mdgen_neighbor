"""Local -> HPC: snapshot, verify, submit, track, fetch.  Tests: tests/test_hpc_run.py.

  python scripts/hpc_run.py submit [--tag T] [--run RUN_ID] [--sbatch=ARG ...] [--env K=V ...] \
      LAUNCHER [LAUNCHER_ARGS ...]
  python scripts/hpc_run.py status RUN_ID
  python scripts/hpc_run.py tail RUN_ID [--job ID] [--err] [--lines N]
  python scripts/hpc_run.py fetch RUN_ID DEST REMOTE_PATH ...

submit uploads a new read-only snapshot (tracked files plus untracked job/ scripts/ mdgen/ hpc/),
checks its SHA-256 manifest remotely, links src/results and src/logs to the run directory, and
submits LAUNCHER from src/ with MDGEN_VENV, MDGEN_RESOURCE_ROOT and MDGEN_RUN_DIR set. --run adds
another job to an existing snapshot after re-checking its manifest. Every submission is appended
to <run>/job_history as "<job id> <comment> <launcher> <args>". REMOTE_PATH is relative to the run
directory, e.g. results/halo_error_49339/halo_error.json or logs/49339.out.
"""
import argparse, datetime, fnmatch, hashlib, json, shlex, subprocess, sys, tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
TARGET = "r15543056@zinfandel.ncree.org"
# Windows OpenSSH times out against this server; ssh and scp run inside WSL.
WSL = ["wsl", "-d", "Ubuntu", "-e"]
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ConnectionAttempts=1",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
RUNS_ROOT = "/share/home/r15543056/MDgen/hpc_runs"
RESOURCE_ROOT = "/share/home/r15543056/MDgen/mdgen"
VENV = "/share/home/r15543056/.venvs/mdgen"
MAX_BYTES = 10 * 2**20
MAX_FETCH_BYTES = 20 * 2**20
EXCLUDE = ["data/*", "splits/*", "results/*", "logs/*", "*.ckpt", "*.pt", "*.pth", "*.safetensors",
           ".env*", "*.pem", "id_rsa*", "id_ed25519*", ".claude/*", ".scratch/*"]
UNTRACKED_DIRS = ("job/", "scripts/", "mdgen/", "hpc/")  # untracked files here are uploaded too


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def wsl_path(path):
    p = PureWindowsPath(Path(path).resolve())
    return f"/mnt/{p.drive[0].lower()}/" + "/".join(p.parts[1:])


def remote(script, timeout=60):
    """Run a bash script on the headnode; the script goes over stdin, so it needs no quoting."""
    # bytes, not text: text-mode stdin on Windows would turn "\n" into "\r\n" for bash
    r = subprocess.run([*WSL, "ssh", *SSH_OPTS, TARGET, "bash -s"], input=script.encode(),
                       capture_output=True, timeout=timeout)
    return subprocess.CompletedProcess(r.args, r.returncode, r.stdout.decode(errors="replace"),
                                       r.stderr.decode(errors="replace"))


def scp(src, dst, timeout=300):
    r = subprocess.run([*WSL, "scp", *SSH_OPTS, src, dst], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode:
        sys.exit(f"scp {src} -> {dst} failed: {r.stderr}")


def select_files():
    names = git("ls-files", "-z").split("\0")
    names += [n for n in git("ls-files", "-z", "--others", "--exclude-standard").split("\0")
              if n.startswith(UNTRACKED_DIRS)]
    keep = []
    for n in sorted(set(filter(None, names))):
        p = ROOT / n
        if not p.is_file() or any(fnmatch.fnmatch(n, g) for g in EXCLUDE):
            continue
        if p.stat().st_size > MAX_BYTES:
            sys.exit(f"too large for snapshot: {n}")
        keep.append(n)
    return keep


def upload(tag):
    files = select_files()
    commit = git("rev-parse", "--short", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip()
                 or git("ls-files", "--others", "--exclude-standard", "--", *UNTRACKED_DIRS).strip())
    run_id = f"{tag}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-{commit}"
    run = f"{RUNS_ROOT}/{run_id}"
    print(f"run_id={run_id} files={len(files)} commit={commit} dirty={dirty}")
    manifest = "".join(f"{hashlib.sha256((ROOT/n).read_bytes()).hexdigest()}  {n}\n" for n in files)
    with tempfile.TemporaryDirectory() as td:
        lst, man, tarball = Path(td, "list"), Path(td, "manifest.sha256"), Path(td, "src.tar")
        lst.write_text("\n".join(files) + "\n", encoding="utf-8")
        man.write_text(manifest, encoding="utf-8", newline="\n")
        subprocess.run(["tar", "-cf", str(tarball), "-T", str(lst)], cwd=ROOT, check=True)
        r = remote(f"set -e; mkdir -p {RUNS_ROOT}; mkdir {run}; mkdir {run}/src {run}/logs {run}/results {run}/cache")
        if r.returncode:
            sys.exit(f"mkdir failed: {r.stderr}")
        scp(wsl_path(tarball), f"{TARGET}:{run}/")
        scp(wsl_path(man), f"{TARGET}:{run}/")
    meta = shlex.quote(json.dumps({"run_id": run_id, "commit": commit, "dirty": dirty}))
    r = remote(f"""set -e
cd {run}
tar -xf src.tar -C src && rm src.tar && mv manifest.sha256 src/
(cd src && sha256sum -c --quiet manifest.sha256)
# launchers write results/ and logs/ relative to the read-only src/
ln -s ../results src/results && ln -s ../logs src/logs
echo {meta} > meta.json
chmod -R a-w src
echo VERIFIED""", timeout=120)
    if "VERIFIED" not in r.stdout:
        sys.exit(f"manifest check failed (not submitting): {r.stdout}{r.stderr}")
    return run_id


def submit_script(run, comment, env, sbatch, argv):
    """Remote bash that submits argv from <run>/src and appends it to <run>/job_history."""
    env = " ".join(shlex.quote(kv) for kv in env)
    sbatch = " ".join(shlex.quote(x) for x in sbatch)
    command = " ".join(shlex.quote(x) for x in argv)
    # printf, not echo "...": inside double quotes the shell would expand $(...) in the args
    return f"""set -e
cd {run}/src
export MDGEN_VENV={VENV} MDGEN_RESOURCE_ROOT={RESOURCE_ROOT} MDGEN_RUN_DIR={run} {env}
job=$(sbatch --parsable --comment={shlex.quote(comment)} --chdir={run}/src \\
  --output={run}/logs/%j.out --error={run}/logs/%j.err {sbatch} {command})
job=${{job%%;*}}
printf '%s %s %s\\n' "$job" {shlex.quote(comment)} {shlex.quote(command)} >> {run}/job_history
echo "JOB=$job"
"""


def submit(a):
    if a.run:
        run_id = a.run
        r = remote(f"cd {RUNS_ROOT}/{run_id}/src && sha256sum -c --quiet manifest.sha256 && echo VERIFIED",
                   timeout=120)
        if "VERIFIED" not in r.stdout:
            sys.exit(f"not submitting: {r.stdout}{r.stderr}")
    else:
        run_id = upload(a.tag)
    run = f"{RUNS_ROOT}/{run_id}"
    comment = f"{run_id}:{a.tag}:{datetime.datetime.now(datetime.timezone.utc):%H%M%S}"
    try:
        r = remote(submit_script(run, comment, a.env, a.sbatch, [a.launcher, *a.args]))
    except subprocess.TimeoutExpired:  # sbatch may still have accepted the job
        r = subprocess.CompletedProcess([], 1, "", "timed out waiting for the headnode\n")
    job = next((line[4:] for line in r.stdout.splitlines() if line.startswith("JOB=")), None)
    if r.returncode or not job:
        print(f"submission_unknown: check `squeue -u $USER -o '%i %k'` for comment={comment} "
              f"before any resubmit\n{r.stdout}{r.stderr}")
        sys.exit(2)
    print(f"submitted job_id={job} run_id={run_id}\nlogs: {run}/logs/{job}.out")


def status(a):
    run = f"{RUNS_ROOT}/{a.run_id}"
    print(remote(f"""cat {run}/job_history
echo '--- queue'
squeue -h -u $USER -o '%i %T %M %R' | grep -wFf <(cut -d' ' -f1 {run}/job_history) || echo none
echo '--- logs'; ls -l {run}/logs; ls {run}/results""").stdout)


def tail(a):
    run = f"{RUNS_ROOT}/{a.run_id}"
    job = a.job or f"$(tail -n 1 {run}/job_history | cut -d' ' -f1)"
    print(remote(f"tail -n {a.lines} {run}/logs/{job}.{'err' if a.err else 'out'}").stdout)


def check_fetch_paths(paths):
    for p in paths:  # stay inside the run directory remotely and inside DEST locally
        pp = PurePosixPath(p)
        if not pp.parts or pp.is_absolute() or ".." in pp.parts or ":" in p or "\\" in p:
            sys.exit(f"fetch: {p!r} must be a relative path inside the run directory")


def fetch(a):
    check_fetch_paths(a.paths)
    run = f"{RUNS_ROOT}/{a.run_id}"
    paths = " ".join(shlex.quote(p) for p in a.paths)
    r = remote(f"set -o pipefail; cd {run} && du -cb -- {paths} | tail -n 1")
    if r.returncode:
        sys.exit(f"fetch: {r.stderr}")
    total = int(r.stdout.split()[0])
    if total > MAX_FETCH_BYTES:
        sys.exit(f"fetch: {total} bytes exceeds {MAX_FETCH_BYTES}; fetch fewer files")
    for p in a.paths:  # keep the run-relative layout so same-named files do not collide
        local = Path(a.dest, p)
        local.parent.mkdir(parents=True, exist_ok=True)
        scp(f"{TARGET}:{run}/{p}", wsl_path(local))
        print(local)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="c", required=True)
    x = s.add_parser("submit")
    x.add_argument("--tag", default="smoke")
    x.add_argument("--run", metavar="RUN_ID", help="add a job to this existing snapshot")
    x.add_argument("--sbatch", action="append", default=[], metavar="ARG",
                   help="extra sbatch argument, e.g. --sbatch=--time=00:30:00")
    x.add_argument("--env", action="append", default=[], metavar="K=V", help="exported for the job")
    x.add_argument("launcher"); x.add_argument("args", nargs=argparse.REMAINDER)
    x.set_defaults(f=submit)
    x = s.add_parser("status"); x.add_argument("run_id"); x.set_defaults(f=status)
    x = s.add_parser("tail"); x.add_argument("run_id"); x.add_argument("--job")
    x.add_argument("--err", action="store_true"); x.add_argument("--lines", type=int, default=100)
    x.set_defaults(f=tail)
    x = s.add_parser("fetch"); x.add_argument("run_id"); x.add_argument("dest"); x.add_argument("paths", nargs="+")
    x.set_defaults(f=fetch)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
