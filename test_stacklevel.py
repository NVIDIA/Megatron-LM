import warnings
import unittest


class TestStackLevel(unittest.TestCase):
    def test_warnings_have_stacklevel(self):
        """Verify that warnings.warn() calls in megatron/core use stacklevel."""
        import ast
        import pathlib

        core = pathlib.Path("megatron/core")
        missing = []
        for f in sorted(core.rglob("*.py")):
            if "__pycache__" in str(f):
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            for m in __import__("re").finditer(r"warnings\.warn\(", text):
                start = m.start()
                depth, i = 0, m.end() - 1
                while i < len(text):
                    if text[i] == "(":
                        depth += 1
                    elif text[i] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                call = text[start : i + 1]
                lineno = text[:start].count("\n") + 1
                if "stacklevel" not in call:
                    missing.append(f"{f}:{lineno}")

        # only module-level import fallbacks should remain without stacklevel
        for m in missing:
            self.assertIn(
                m.split("/")[-1].split(":")[0],
                ["__init__.py", "clip_grads.py", "optimizer.py", "hyper_comm_grid.py"],
                f"Missing stacklevel in {m}",
            )


if __name__ == "__main__":
    unittest.main()
