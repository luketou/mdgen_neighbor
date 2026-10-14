"""Fast source/schema checks; no rendering, model imports or GPU required."""
import ast
import hashlib
import json
from pathlib import Path
import re
import unittest

import build_tour as build


class TourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((build.HERE/'tour-data.json').read_text())

    def test_every_in_scope_file_is_indexed(self):
        paths,_=build.git_paths()
        self.assertEqual(set(paths),{f['path'] for f in self.data['files']})
        self.assertEqual(len(paths),len(self.data['files']))

    def test_source_hashes_are_fresh(self):
        for f in self.data['files']:
            self.assertEqual(f['sha256'],hashlib.sha256((build.ROOT/f['path']).read_bytes()).hexdigest(),f['path'])

    def test_all_anchors_steps_edges_and_chinese_labels(self):
        build.validate(self.data['chapters']+self.data['file_pages'])

    def test_all_symbols_appear_on_file_pages(self):
        pages={c['id']:c for c in self.data['file_pages']}
        for f in self.data['files']:
            self.assertTrue(f['pages'],f['path'])
            shown=[n['label'] for p in f['pages'] for n in pages[p]['nodes']]
            if f['symbols']:self.assertEqual(shown,[s['name'] for s in f['symbols']],f['path'])

    def test_all_named_python_definitions_including_nested_are_indexed(self):
        for f in self.data['files']:
            if not f['path'].endswith('.py'):continue
            tree=ast.parse((build.ROOT/f['path']).read_text())
            expected=sum(isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) for n in ast.walk(tree))
            self.assertEqual(expected,len(f['symbols']),f['path'])

    def test_no_diagram_shape_keywords_in_tensor_fields(self):
        for c in self.data['chapters']:
            for n in c['nodes']:
                self.assertNotIn(n['shape'],['process','cube','table','artifact','decision','database'])

    def test_generated_html_injects_only_data_script(self):
        html=(build.HERE/'index.html').read_text()
        raw=re.search(r'<script id="tour-data" type="application/json">(.*?)</script>',html,re.S).group(1)
        self.assertEqual(json.loads(raw)['coverage'],self.data['coverage'])
        self.assertNotIn('__TOUR_DATA__',raw)

    def test_local_imports_refer_to_real_files(self):
        paths={f['path'] for f in self.data['files']}
        for f in self.data['files']:
            for p in f['imports']:self.assertIn(p,paths)

    def test_source_and_renderer_are_syntax_valid(self):
        for path in build.HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))


if __name__=='__main__':unittest.main(verbosity=2)
