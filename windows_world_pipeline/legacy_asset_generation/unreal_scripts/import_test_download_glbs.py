from __future__ import annotations

import unreal


GLB_IMPORTS = [
    ("C:/Users/user/Downloads/test1.glb", "/Game/Generated/ImportTest", "SM_test1"),
    ("C:/Users/user/Downloads/test2.glb", "/Game/Generated/ImportTest", "SM_test2"),
    ("C:/Users/user/Downloads/test3.glb", "/Game/Generated/ImportTest", "SM_test3"),
]


def import_glb(glb_path: str, destination_path: str, asset_name: str) -> None:
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
        unreal.log_error(f"[cosmos] no asset imported from {glb_path}")
        return

    unreal.log(f"[cosmos] imported {glb_path} -> {imported_paths[0]}")


def main() -> None:
    for glb_path, destination_path, asset_name in GLB_IMPORTS:
        import_glb(glb_path, destination_path, asset_name)
    unreal.EditorAssetLibrary.save_directory("/Game/Generated/ImportTest")
    unreal.log("[cosmos] import test finished")


main()
