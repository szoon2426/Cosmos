# -*- coding: utf-8 -*-
"""
Unreal Engine Remote Control API 탐색 스크립트
- GET /remote/presets : 프리셋 목록 조회
- GET /remote/preset/{name} : 프리셋 상세 (ExposedProperties 확인)
- GET /remote/preset/{name}/property/{DisplayName} : 프로퍼티 값 읽기
- PUT /remote/preset/{name}/property/{DisplayName} : 프로퍼티 값 쓰기
"""

import requests
import json
import sys
import io

# Windows cp949 인코딩 문제 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


BASE_URL = "http://localhost:30010"
HEADERS = {"Content-Type": "application/json"}


def pretty(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def step1_list_presets():
    """1단계: 모든 프리셋 목록 조회"""
    print("\n" + "=" * 60)
    print("[STEP 1] GET /remote/presets")
    print("=" * 60)
    try:
        r = requests.get(f"{BASE_URL}/remote/presets", headers=HEADERS, timeout=5)
        print(f"Status: {r.status_code}")
        data = r.json()
        pretty(data)

        presets = data.get("Presets", [])
        names = [p.get("Name", p.get("name", "???")) for p in presets]
        print(f"\n>> 발견된 프리셋: {names}")
        return names, data
    except requests.ConnectionError:
        print("[ERROR] 연결 실패!")
        print("  - 언리얼 에디터가 실행 중인지 확인")
        print("  - Remote Control API 플러그인이 활성화되어 있는지 확인")
        print(f"  - 포트가 맞는지 확인 (현재: {BASE_URL})")
        return [], None
    except Exception as e:
        print(f"[ERROR] {e}")
        return [], None


def step2_get_preset(preset_name):
    """2단계: 특정 프리셋 상세 정보 조회"""
    print("\n" + "=" * 60)
    print(f"[STEP 2] GET /remote/preset/{preset_name}")
    print("=" * 60)
    try:
        r = requests.get(f"{BASE_URL}/remote/preset/{preset_name}", headers=HEADERS, timeout=5)
        print(f"Status: {r.status_code}")
        data = r.json()
        pretty(data)

        preset_data = data.get("Preset", data)

        # ExposedProperties는 Groups 안에 중첩되어 있음
        groups = preset_data.get("Groups", [])
        display_names = []

        for group in groups:
            group_name = group.get("Name", "N/A")
            exposed = group.get("ExposedProperties", [])
            exposed_funcs = group.get("ExposedFunctions", [])

            print(f"\n>> Group: \"{group_name}\" | Properties: {len(exposed)} | Functions: {len(exposed_funcs)}")
            print("-" * 40)

            for i, prop in enumerate(exposed):
                dn = prop.get("DisplayName", "N/A")
                uid = prop.get("ID", "N/A")
                underlying = prop.get("UnderlyingProperty", {})
                prop_name = underlying.get("Name", "N/A") if isinstance(underlying, dict) else "N/A"
                prop_type = underlying.get("Type", "N/A") if isinstance(underlying, dict) else "N/A"
                metadata = prop.get("Metadata", {})
                owners = prop.get("OwnerObjects", [])
                owner_name = owners[0].get("Name", "N/A") if owners else "N/A"

                display_names.append(dn)
                print(f"  [{i}] DisplayName  = \"{dn}\"")
                print(f"       ID            = \"{uid}\"")
                print(f"       PropertyName  = \"{prop_name}\"")
                print(f"       Type          = \"{prop_type}\"")
                print(f"       Owner         = \"{owner_name}\"")
                if metadata.get("Min") or metadata.get("Max"):
                    print(f"       Range         = [{metadata.get('Min','?')} ~ {metadata.get('Max','?')}]")
                print()

            if exposed_funcs:
                for i, func in enumerate(exposed_funcs):
                    fn = func.get("DisplayName", func.get("Name", "N/A"))
                    print(f"  [F{i}] Function: \"{fn}\"")
                print()

        return display_names, data
    except Exception as e:
        print(f"[ERROR] {e}")
        return [], None


def step3_get_property(preset_name, display_name):
    """3단계: 프로퍼티 값 읽기 (GET)"""
    print("\n" + "-" * 60)
    print(f"[STEP 3] GET /remote/preset/{preset_name}/property/{display_name}")
    print("-" * 60)
    try:
        r = requests.get(
            f"{BASE_URL}/remote/preset/{preset_name}/property/{display_name}",
            headers=HEADERS, timeout=5
        )
        print(f"Status: {r.status_code}")
        data = r.json()
        pretty(data)
        return data
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def step4_put_property(preset_name, display_name, value):
    """4단계: 프로퍼티 값 쓰기 (PUT)"""
    print("\n" + "-" * 60)
    print(f"[STEP 4] PUT /remote/preset/{preset_name}/property/{display_name}")
    print("-" * 60)

    body = {"PropertyValue": value}
    print(f"Body: {json.dumps(body, indent=2, ensure_ascii=False)}")

    try:
        r = requests.put(
            f"{BASE_URL}/remote/preset/{preset_name}/property/{display_name}",
            headers=HEADERS,
            json=body,
            timeout=5
        )
        print(f"Status: {r.status_code}")
        if r.text:
            try:
                pretty(r.json())
            except:
                print(f"Response: {r.text}")
        return r.status_code
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def main():
    print("=" * 60)
    print("  Unreal Engine Remote Control API Test")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)

    # STEP 1: 프리셋 목록
    preset_names, raw = step1_list_presets()
    if not preset_names:
        print("\n[!] 프리셋이 없거나 연결에 실패했습니다.")
        return

    # STEP 2: 각 프리셋 상세 조회
    all_info = {}
    for name in preset_names:
        display_names, _ = step2_get_preset(name)
        all_info[name] = display_names

    # 전체 요약
    print("\n" + "=" * 60)
    print("  [SUMMARY] 전체 프리셋 / 프로퍼티 요약")
    print("=" * 60)
    for pname, dnames in all_info.items():
        print(f"\n  Preset: \"{pname}\"")
        for dn in dnames:
            print(f"    -> DisplayName: \"{dn}\"")

    # STEP 3: 모든 프로퍼티 GET 테스트
    print("\n" + "=" * 60)
    print("  [STEP 3] 각 프로퍼티 값 GET 테스트")
    print("=" * 60)
    for pname, dnames in all_info.items():
        for dn in dnames:
            step3_get_property(pname, dn)

    # STEP 4: PUT 테스트 (기본 비활성화)
    print("\n" + "=" * 60)
    print("  [STEP 4] PUT 테스트")
    print("=" * 60)
    print("  PUT은 값을 변경하므로, 아래 주석을 해제하고 직접 실행하세요.")
    print("  예시:")
    print('    step4_put_property("MyPreset", "MyFloat", 0.5)')
    print('    step4_put_property("MyPreset", "MyColor", {"R":1,"G":0,"B":0,"A":1})')
    print('    step4_put_property("MyPreset", "MyVector", {"X":100,"Y":200,"Z":300})')

    # === PUT 테스트하려면 아래 주석 해제 후 값 수정 ===
    # step4_put_property("프리셋이름", "DisplayName", 원하는값)


if __name__ == "__main__":
    main()
