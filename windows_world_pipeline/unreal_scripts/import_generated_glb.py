from __future__ import annotations

import sys

import unreal


def parse_arg(prefix: str) -> str:
    for arg in sys.argv:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    raise RuntimeError(f"Missing required argument: {prefix}")


def import_glb(glb_path: str, destination_path: str, asset_name: str) -> str:
    task = unreal.AssetImportTask()
    task.filename = glb_path
    task.destination_path = destination_path
    task.destination_name = asset_name
    task.automated = True
    task.replace_existing = True
    task.save = True

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported_paths = list(task.imported_object_paths)
    if not imported_paths:
        raise RuntimeError(f"No asset imported from {glb_path}")

    unreal.EditorAssetLibrary.save_directory(destination_path)
    return imported_paths[0]


def main() -> None:
    glb_path = parse_arg("-glb=")
    destination_path = parse_arg("-dest=")
    asset_name = parse_arg("-name=")
    imported_path = import_glb(glb_path, destination_path, asset_name)
    unreal.log(f"[cosmos] imported generated GLB -> {imported_path}")


if __name__ == "__main__":
    main()
