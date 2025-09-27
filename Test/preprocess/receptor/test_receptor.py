# tests/test_receptor.py
import json
import tempfile
import unittest
from pathlib import Path

from prodock.preprocess.receptor import receptor, repr_helpers


class TestReceptor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # save originals to restore later
        self._orig_fix_pdb = receptor.fix_pdb
        self._orig_min_openmm = receptor.minimize_with_openmm
        self._orig_min_obabel = receptor.minimize_with_obabel
        self._orig_conv_ob = receptor.convert_with_obabel
        self._orig_conv_mk = receptor.convert_with_mekoo
        self._orig_cmd = getattr(receptor, "cmd", None)

    def tearDown(self):
        receptor.fix_pdb = self._orig_fix_pdb
        receptor.minimize_with_openmm = self._orig_min_openmm
        receptor.minimize_with_obabel = self._orig_min_obabel
        receptor.convert_with_obabel = self._orig_conv_ob
        receptor.convert_with_mekoo = self._orig_conv_mk
        if self._orig_cmd is not None:
            receptor.cmd = self._orig_cmd

        try:
            for p in self.tmpdir.rglob("*"):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            self.tmpdir.rmdir()
        except Exception:
            pass

    def test_expected_output_for_variants(self):
        rp = receptor.ReceptorPrep(use_meeko=True)
        inp = self.tmpdir / "protein.pdb"
        inp.write_text("ATOM\n")
        out1 = rp.expected_output_for(
            inp, self.tmpdir, out_fmt="pdbqt", add_prep_suffix=True, basename=None
        )
        self.assertTrue(str(out1).endswith("_prep.pdbqt"))

        out2 = rp.expected_output_for(
            inp, self.tmpdir, out_fmt=".pdb", add_prep_suffix=False, basename="custom"
        )
        self.assertTrue(str(out2).endswith("custom.pdb"))

    def test_save_report_writes_json(self):
        rp = receptor.ReceptorPrep(use_meeko=False)
        rp._last_simulation_report = {"final_artifact": "foo", "out_fmt": "pdb"}
        target = self.tmpdir / "report.json"
        written = rp.save_report(target)
        self.assertTrue(target.exists())
        data = json.loads(target.read_text())
        self.assertEqual(data["final_artifact"], "foo")
        self.assertEqual(str(written), str(target))

    def test_postprocess_pymol_uses_cmd(self):
        called = {}

        class FakeCmd:
            def load(self, *a, **k):
                called["load"] = True

            def alter(self, *a, **k):
                called["alter"] = True

            def select(self, *a, **k):
                called.setdefault("select", []).append(a)

            def remove(self, *a, **k):
                called["remove"] = True

            def save(self, *a, **k):
                called["save"] = True

            def delete(self, *a, **k):
                called["delete"] = True

        receptor.cmd = FakeCmd()
        p = self.tmpdir / "proc.pdb"
        p.write_text("ATOM\n")
        # This uses the receptor.ReceptorPrep._postprocess_pymol signature (path inputs)
        receptor.ReceptorPrep()._postprocess_pymol(p, start_at=2, cofactors=["HOH"])
        self.assertIn("load", called)
        self.assertIn("save", called)

    def test_prep_fallback_to_obabel_creates_pdbqt(self):
        # stub low-level helpers directly on module (no unittest.mock)
        def stub_fix_pdb(path):
            return "FIXED_SENTINEL"

        def stub_min_openmm(mod, **kw):
            raise RuntimeError("openmm not available")

        def stub_min_obabel(inp, out, steps=100):
            out.write_text("MINIMIZED_PDB\n")
            return out

        def stub_conv_obabel(inp, out, extra_args=None):
            out.write_text("CONVERTED_PDBQT\n")
            return None

        receptor.fix_pdb = stub_fix_pdb
        receptor.minimize_with_openmm = stub_min_openmm
        receptor.minimize_with_obabel = stub_min_obabel
        receptor.convert_with_obabel = stub_conv_obabel

        inp = self.tmpdir / "inp.pdb"
        inp.write_text("ATOM\n")
        rp = receptor.ReceptorPrep(use_meeko=True)
        rp.prep(str(inp), str(self.tmpdir), out_fmt="pdbqt", add_prep_suffix=True)

        self.assertTrue(rp.used_obabel)
        self.assertEqual(rp.minimized_stage, "obabel")
        self.assertIsNotNone(rp.final_artifact)
        self.assertTrue(str(rp.final_artifact).endswith(".pdbqt"))
        rpt = rp.last_simulation_report
        self.assertIsInstance(rpt, dict)
        self.assertIn("used_obabel", rpt)
        self.assertTrue(rpt["used_obabel"])

    def test_repr_mixin_behaviour(self):
        class D(repr_helpers.ReprMixin):
            pass

        d = D()
        d._final_artifact = None
        d._last_simulation_report = None
        d._used_obabel = False
        r = repr(d)
        self.assertIsInstance(r, str)
        self.assertIn("ReceptorPrep Summary", r)

        d._last_simulation_report = {
            "out_fmt": "pdbqt",
            "mekoo_info": {"produced": ["x.pdbqt"]},
        }
        info = d._repr_basic_info()
        self.assertEqual(d._repr_converter_status(info), "mekoo")

        d._used_obabel = True
        info2 = d._repr_basic_info()
        self.assertEqual(d._repr_converter_status(info2), "OpenBabel")


if __name__ == "__main__":
    unittest.main()
