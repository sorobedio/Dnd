import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from workspace.dnd_downstream.evaluate import (membership, report, select_originals, spread, summarize,
                                               training_selection, utilization_from_memory)

SETTINGS = dict(datasets=["ARC-e", "BoolQ"], dataset_tag="ARC-c", real_length=2)


def checkpoints(folder, steps, extra=()):
    folder.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (folder / f"{step}.safetensors").write_text(str(step))
    for name in extra:
        (folder / name).write_text(name)
    return folder


class SelectionTests(unittest.TestCase):
    def test_takes_the_highest_steps_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = checkpoints(Path(directory), [50, 300, 100, 250, 200, 150], extra=["last.safetensors"])
            self.assertEqual([p.stem for p in select_originals(folder, 5)], ["100", "150", "200", "250", "300"])

    def test_rejects_a_folder_with_too_few_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = checkpoints(Path(directory), [100, 200])
            with self.assertRaises(ValueError):
                select_originals(folder, 5)

    def test_membership_distinguishes_seen_and_unseen_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints(root / "ARC-e", [100, 200, 300])
            checkpoints(root / "BoolQ", [100, 200])
            selection = training_selection(root, SETTINGS)
            seen = sorted(selection["ARC-e"])[0]
            self.assertEqual(membership("ARC-e", seen, SETTINGS, selection), "train")
            unseen = next(p for p in (root / "ARC-e").glob("*.safetensors") if str(p.resolve()) not in selection["ARC-e"])
            self.assertEqual(membership("ARC-e", unseen, SETTINGS, selection), "unseen_checkpoint")
            self.assertEqual(membership("ARC-c", unseen, SETTINGS, selection), "held_out_task")
            self.assertEqual(membership("OBQA", unseen, SETTINGS, selection), "unseen_task")

    def test_training_selection_requires_the_full_training_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints(root / "ARC-e", [100])
            checkpoints(root / "BoolQ", [100, 200])
            with self.assertRaises(ValueError):
                training_selection(root, SETTINGS)


class SummaryTests(unittest.TestCase):
    def test_spread_of_a_single_value_has_no_deviation(self):
        self.assertEqual(spread([0.5]), dict(mean=0.5, std=0.0, min=0.5, max=0.5))

    def test_summary_compares_generated_samples_with_the_originals(self):
        row = dict(reference="original_300", variants=[
            dict(name="dnd_0", kind="dnd", accuracy=0.5, relative_l2=0.1, cosine=0.9),
            dict(name="dnd_1", kind="dnd", accuracy=0.7, relative_l2=0.3, cosine=0.7),
            dict(name="original_200", kind="original", accuracy=0.4),
            dict(name="original_300", kind="original", accuracy=0.8),
        ])
        predictions = {
            "dnd_0": [dict(answer="A"), dict(answer="B")],
            "dnd_1": [dict(answer="A"), dict(answer="A")],
            "original_200": [dict(answer="B"), dict(answer="B")],
            "original_300": [dict(answer="A"), dict(answer="B")],
        }
        summary = summarize(row, predictions)
        self.assertAlmostEqual(summary["dnd_mean"], 0.6)
        self.assertAlmostEqual(summary["original_mean"], 0.6)
        self.assertAlmostEqual(summary["delta_percentage_points"], 0.0)
        # dnd_0 matches both answers, dnd_1 matches one of two.
        self.assertAlmostEqual(summary["dnd_agreement_with_reference"], 0.75)
        self.assertAlmostEqual(summary["original_agreement_with_reference"], 0.5)
        self.assertAlmostEqual(summary["dnd_relative_l2"], 0.2)
        self.assertAlmostEqual(summary["dnd_cosine"], 0.8)

    def test_summary_without_other_originals_reports_full_agreement(self):
        row = dict(reference="original_300", variants=[
            dict(name="dnd_0", kind="dnd", accuracy=0.5, relative_l2=0.1, cosine=0.9),
            dict(name="original_300", kind="original", accuracy=0.8),
        ])
        predictions = {"dnd_0": [dict(answer="A")], "original_300": [dict(answer="A")]}
        self.assertEqual(summarize(row, predictions)["original_agreement_with_reference"], 1.0)


class ReportTests(unittest.TestCase):
    def test_report_renders_both_tables(self):
        task = dict(task="ARC-c", n=1172, reference="original_250", dnd_mean=0.51, dnd_std=0.01,
                    original_mean=0.56, original_std=0.002, delta_percentage_points=-5.0,
                    dnd_agreement_with_reference=0.6, original_agreement_with_reference=0.99,
                    variants=[dict(name="dnd_0", kind="dnd", accuracy=0.51, invalid=0, truncated=0,
                                   relative_l2=0.42, cosine=0.91),
                              dict(name="original_250", kind="original", accuracy=0.56, invalid=1,
                                   truncated=2, dnd_membership="held_out_task")])
        result = dict(protocol="Greedy", max_new_tokens=1024, dnd_checkpoint="checkpoints/x.pth",
                      dnd_train_tasks=["ARC-e"], dnd_held_out_task="ARC-c", samples=5, originals=5,
                      tasks=[task])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "results.json").write_text(json.dumps(result))
            report(Namespace(output=str(output), markdown=str(output / "report.md")))
            text = (output / "report.md").read_text()
        self.assertIn("| ARC-c | held_out_task | 1172 | 56.00% ± 0.20 | 51.00% ± 1.00 | -5.00 |", text)
        self.assertIn("| ARC-c | dnd_0 | generated | 51.00% | 0/1172 | 0/1172 | 42.00% | 0.9100 |", text)
        self.assertIn("| ARC-c | original_250 | held_out_task | 56.00% | 1/1172 | 2/1172 | — | — |", text)


if __name__ == "__main__":
    unittest.main()


class UtilizationTests(unittest.TestCase):
    def test_sizes_to_the_free_memory_with_headroom(self):
        fraction, message = utilization_from_memory(None, (40960, 81920))
        self.assertEqual(fraction, 0.47)
        self.assertIn("40960 MiB free of 81920 MiB", message)

    def test_caps_the_fraction_on_an_idle_gpu(self):
        self.assertEqual(utilization_from_memory(None, (81000, 81920))[0], 0.85)

    def test_rejects_a_request_that_does_not_fit(self):
        with self.assertRaises(RuntimeError) as raised:
            utilization_from_memory(0.6, (20480, 81920))
        self.assertIn("only 20480 MiB", str(raised.exception))

    def test_honours_a_request_that_fits(self):
        self.assertEqual(utilization_from_memory(0.2, (20480, 81920))[0], 0.2)

    def test_refuses_a_gpu_with_almost_nothing_free(self):
        with self.assertRaises(RuntimeError):
            utilization_from_memory(None, (4096, 81920))

    def test_falls_back_when_the_gpu_cannot_be_queried(self):
        self.assertEqual(utilization_from_memory(None, None)[0], 0.25)
        self.assertEqual(utilization_from_memory(0.4, None)[0], 0.4)
