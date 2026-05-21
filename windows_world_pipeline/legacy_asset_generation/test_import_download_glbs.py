from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_UNREAL_EDITOR_CMD = "C:/Program Files/Epic Games/UE_5.6/Engine/Binaries/Win64/UnrealEditor-Cmd.exe"
DEFAULT_UPROJECT = "D:/Unreal_Projects/GG/GG.uproject"
DEFAULT_IMPORT_SCRIPT = Path(__file__).resolve().parent / "unreal_scripts" / "import_generated_glb.py"
DEFAULT_DESTINATION = "/Game/Generated/ImportTest"
DEFAULT_GLB_PATHS = [
    "C:/Users/user/Downloads/test1.glb",
]
ALL_GLB_PATHS = [
    "C:/Users/user/Downloads/test1.glb",
    "C:/Users/user/Downloads/test2.glb",
    "C:/Users/user/Downloads/test3.glb",
]


def asset_name_from_path(path: Path) -> str:
    return f"SM_{path.stem}"


def run_import(
    unreal_editor_cmd: str,
    uproject: str,
    import_script: Path,
    glb_path: Path,
    destination: str,
    asset_name: str,
    dry_run: bool,
) -> None:
    command = [
        unreal_editor_cmd,
        uproject,
        f"-ExecutePythonScript={import_script.as_posix()}",
        f"-glb={glb_path.as_posix()}",
        f"-dest={destination}",
        f"-name={asset_name}",
    ]

    print("[import-test] command:")
    print("  " + "\n  ".join(command))

    if dry_run:
        return

    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import test GLB files into Unreal automatically.")
    parser.add_argument("--unreal-editor-cmd", default=DEFAULT_UNREAL_EDITOR_CMD)
    parser.add_argument("--uproject", default=DEFAULT_UPROJECT)
    parser.add_argument("--import-script", default=str(DEFAULT_IMPORT_SCRIPT))
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--glb", action="append", default=None, help="GLB path. Can be passed multiple times.")
    parser.add_argument("--all", action="store_true", help="Import test1.glb, test2.glb, and test3.glb.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import_script = Path(args.import_script).resolve()
    default_paths = ALL_GLB_PATHS if args.all else DEFAULT_GLB_PATHS
    glb_paths = [Path(path).resolve() for path in (args.glb or default_paths)]

    missing = [path for path in glb_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"[import-test] missing -> {path}")
        raise SystemExit(1)

    if not import_script.exists():
        print(f"[import-test] missing import script -> {import_script}")
        raise SystemExit(1)

    for glb_path in glb_paths:
        run_import(
            args.unreal_editor_cmd,
            args.uproject,
            import_script,
            glb_path,
            args.destination,
            asset_name_from_path(glb_path),
            args.dry_run,
        )

    print(f"[import-test] done -> {len(glb_paths)} file(s)")


if __name__ == "__main__":
    main()
