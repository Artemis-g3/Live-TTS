from __future__ import annotations

import sys
from pathlib import Path


def add_code_root_to_path() -> None:
    if getattr(sys, "frozen", False):
        project_root = Path(sys.executable).resolve().parent
        if project_root.name.lower() == "dist":
            project_root = project_root.parent
    else:
        project_root = Path(__file__).resolve().parents[1]
    code_root = project_root / "code"
    for path in [project_root, code_root]:
        if path.exists():
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)


add_code_root_to_path()

from PyQt6.QtWidgets import QApplication

from GUI.ui.main_window import MainWindow
from GUI.ui.main_window import run_app


def main() -> int:
    if "--smoke-test" in sys.argv:
        app = QApplication([])
        window = MainWindow()
        print(window.windowTitle())
        print(",".join(window.role_names))
        window.close()
        app.quit()
        return 0
    return run_app()


if __name__ == "__main__":
    sys.exit(main())
