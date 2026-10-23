"""Check tracked text line endings, local links in agent navigation docs, and that launchers
match job/env.sh's GPU preflight to their partition."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


def launcher_errors(job_dir):
    """job/env.sh requires a GPU unless MDGEN_REQUIRE_GPU=0, which only CPU partitions may set."""
    errors = []
    for launcher in sorted(job_dir.glob('*.sh')):
        text = launcher.read_text(encoding='utf-8')
        partition = re.search(r'^#SBATCH --partition=(\S+)', text, re.M)
        if partition is None or 'job/env.sh' not in text:
            continue
        cpu = partition.group(1).startswith('cpu')
        opted_out = re.search(r'\bMDGEN_REQUIRE_GPU=0\b', text) is not None
        if cpu and not opted_out:
            errors.append(f'{launcher.name}: CPU partition {partition.group(1)} needs '
                          'MDGEN_REQUIRE_GPU=0 before sourcing job/env.sh')
        elif opted_out and not cpu:
            errors.append(f'{launcher.name}: MDGEN_REQUIRE_GPU=0 skips the GPU preflight on '
                          f'GPU partition {partition.group(1)}')
    return errors


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    records = subprocess.check_output(
        ['git', 'ls-files', '--eol', '-z'], cwd=root).decode().split('\0')
    for record in filter(None, records):
        attributes, name = record.split('\t', 1)
        if 'w/crlf' in attributes or 'w/mixed' in attributes:
            errors.append(f'{name}: tracked text must use LF line endings')

    sources = [root / 'AGENTS.md', *sorted((root / 'docs/agents').glob('*.md'))]
    for source in sources:
        if not source.is_file():
            errors.append(f'{source.relative_to(root)}: navigation file is missing')
            continue
        fenced = False
        for number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
            if line.lstrip().startswith('```'):
                fenced = not fenced
            if fenced:
                continue
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', line):
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                destination = source.parent / unquote(url.path)
                if not destination.exists():
                    errors.append(f'{source.relative_to(root)}:{number}: missing link target {target}')
    errors += launcher_errors(root / 'job')
    for error in errors:
        print(error)
    if not errors:
        print('Repository checks passed: LF line endings, agent navigation links and launcher GPU preflight.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
