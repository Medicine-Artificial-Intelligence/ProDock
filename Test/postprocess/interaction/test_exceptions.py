from __future__ import annotations

import unittest

from prodock.postprocess.interaction.exceptions import (
    InteractionError,
    InteractionProcessingError,
    InvalidLigandInputError,
    MissingDependencyError,
    VisualizationError,
)


class TestInteractionExceptions(unittest.TestCase):
    def test_interaction_error_is_runtime_error(self) -> None:
        exc = InteractionError("base failure")
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(str(exc), "base failure")

    def test_missing_dependency_error_inheritance(self) -> None:
        exc = MissingDependencyError("missing package")
        self.assertIsInstance(exc, MissingDependencyError)
        self.assertIsInstance(exc, InteractionError)
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(str(exc), "missing package")

    def test_invalid_ligand_input_error_inheritance(self) -> None:
        exc = InvalidLigandInputError("bad ligand")
        self.assertIsInstance(exc, InvalidLigandInputError)
        self.assertIsInstance(exc, InteractionError)
        self.assertEqual(str(exc), "bad ligand")

    def test_visualization_error_inheritance(self) -> None:
        exc = VisualizationError("draw failed")
        self.assertIsInstance(exc, VisualizationError)
        self.assertIsInstance(exc, InteractionError)
        self.assertEqual(str(exc), "draw failed")

    def test_interaction_processing_error_inheritance(self) -> None:
        exc = InteractionProcessingError("batch failed")
        self.assertIsInstance(exc, InteractionProcessingError)
        self.assertIsInstance(exc, InteractionError)
        self.assertEqual(str(exc), "batch failed")

    def test_catch_base_class_for_all_custom_errors(self) -> None:
        errors = [
            MissingDependencyError("dep"),
            InvalidLigandInputError("ligand"),
            VisualizationError("viz"),
            InteractionProcessingError("process"),
        ]

        for err in errors:
            with self.subTest(error_type=type(err).__name__):
                try:
                    raise err
                except InteractionError as caught:
                    self.assertIs(caught, err)

    def test_exception_chaining_example(self) -> None:
        try:
            try:
                raise ImportError("No module named 'prolif'")
            except ImportError as exc:
                raise MissingDependencyError("prolif is required") from exc
        except MissingDependencyError as caught:
            self.assertEqual(str(caught), "prolif is required")
            self.assertIsInstance(caught.__cause__, ImportError)
            self.assertIn("prolif", str(caught.__cause__))


if __name__ == "__main__":
    unittest.main()
