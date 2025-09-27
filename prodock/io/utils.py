import sys
import importlib
from typing import Dict, Any


def shutdown_pymol(remove_modules: bool = True, quiet: bool = True) -> Dict[str, Any]:
    """
    Try to shut down any already-imported / running PyMOL instance and optionally
    remove pymol-related modules from sys.modules so a fresh import (or fake
    module injection) is possible.

    This is best-effort: it will swallow errors and report them in the returned dict.

    Args:
        remove_modules: if True, delete any sys.modules entries that equal
                        "pymol" or start with "pymol." after shutdown attempts.
        quiet: if False, print a short summary; default True.

    Returns:
        A dict with keys:
          - imported: bool, whether an import of "pymol" succeeded initially
          - actions: dict mapping attempted action -> (success: bool, message:str)
          - clean: bool, whether no "pymol" entries remain in sys.modules
    """
    res: Dict[str, Any] = {"imported": False, "actions": {}, "clean": None}

    # Try to import existing pymol module (if any)
    try:
        pymol = importlib.import_module("pymol")
        res["imported"] = True
    except ModuleNotFoundError:
        pymol = None
        res["imported"] = False
    except Exception as exc:
        pymol = None
        res["actions"]["import_error"] = (False, f"import raised: {exc!r}")

    # If imported, try graceful shutdown options (best-effort)
    if pymol is not None:
        # Common APIs: cmd.quit(), cmd.reinitialize(), cmd.delete("all")
        try:
            if hasattr(pymol, "cmd") and hasattr(pymol.cmd, "quit"):
                try:
                    pymol.cmd.quit()
                    res["actions"]["cmd.quit"] = (True, "called cmd.quit()")
                except Exception as exc:
                    res["actions"]["cmd.quit"] = (False, f"raised: {exc!r}")
        except Exception as exc:
            res["actions"]["cmd.quit"] = (False, f"checking/calling raised: {exc!r}")

        try:
            if hasattr(pymol, "cmd") and hasattr(pymol.cmd, "reinitialize"):
                try:
                    pymol.cmd.reinitialize()
                    res["actions"]["cmd.reinitialize"] = (
                        True,
                        "called cmd.reinitialize()",
                    )
                except Exception as exc:
                    res["actions"]["cmd.reinitialize"] = (False, f"raised: {exc!r}")
        except Exception as exc:
            res["actions"]["cmd.reinitialize"] = (
                False,
                f"checking/calling raised: {exc!r}",
            )

        try:
            if hasattr(pymol, "cmd") and hasattr(pymol.cmd, "delete"):
                try:
                    pymol.cmd.delete("all")
                    res["actions"]["cmd.delete_all"] = (
                        True,
                        "called cmd.delete('all')",
                    )
                except Exception as exc:
                    res["actions"]["cmd.delete_all"] = (False, f"raised: {exc!r}")
        except Exception as exc:
            res["actions"]["cmd.delete_all"] = (
                False,
                f"checking/calling raised: {exc!r}",
            )

        # Some builds expose other helpers; call conservatively
        try:
            if hasattr(pymol, "stop"):
                try:
                    pymol.stop()
                    res["actions"]["pymol.stop"] = (True, "called pymol.stop()")
                except Exception as exc:
                    res["actions"]["pymol.stop"] = (False, f"raised: {exc!r}")
        except Exception as exc:
            res["actions"]["pymol.stop"] = (False, f"checking/calling raised: {exc!r}")

    # Remove any cached pymol modules so future import returns a fresh module
    if remove_modules:
        removed = []
        failed = []
        for name in list(sys.modules):
            if name == "pymol" or name.startswith("pymol."):
                try:
                    del sys.modules[name]
                    removed.append(name)
                except Exception as exc:
                    failed.append((name, repr(exc)))
        res["actions"]["removed_sys_modules"] = (
            len(removed) > 0,
            f"removed={removed}, failed={failed}",
        )

    # Final cleanliness check
    still = [m for m in sys.modules if m == "pymol" or m.startswith("pymol.")]
    res["clean"] = len(still) == 0

    if not quiet:
        import pprint

        pprint.pprint(res)

    return res
