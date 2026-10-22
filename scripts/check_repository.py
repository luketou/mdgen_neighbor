"""Check tracked text line endings and local links in agent navigation docs."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


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
    for error in errors:
        print(error)
    if not errors:
        print('Repository checks passed: LF line endings and agent navigation links.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
