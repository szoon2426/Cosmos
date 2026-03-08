"""
ue_bridge.py — Python -> Unreal Engine Remote Control API bridge

NiagaraActor_2의 Blueprint 함수 SetFountainVelocity를 호출합니다.
포트: 30010
"""

import threading
import requests


# -- Settings ------------------------------------------------------------
UE_HOST = "http://127.0.0.1"
UE_PORT = 30010
# 에디터 원본(L_01) 대신 플레이 중인 런타임 월드(UEDPIE_0_L_01)를 타겟팅합니다.
ACTOR_PATH = "/Game/Test/UEDPIE_0_L_01.L_01:PersistentLevel.NS_Fountain_Blueprint_C_0"
STATUE_ACTOR_PATH = "/Game/Test/UEDPIE_0_L_01.L_01:PersistentLevel.NewBlueprint_C_1"

BASE_URL = f"{UE_HOST}:{UE_PORT}"
CALL_URL = f"{BASE_URL}/remote/object/call"
# ------------------------------------------------------------------------


class UEBridge:
    def __init__(self):
        pass

    def start(self):
        print(f"[UEBridge] Remote Control API -> {BASE_URL}")
        print(f"[UEBridge] Target: {ACTOR_PATH}")

    def send(self, payload: dict):
        fountain_thread = threading.Thread(
            target=self._send_fountain,
            args=(payload.get("fountain", {}),),
            daemon=True,
        )
        fountain_thread.start()

        statue_thread = threading.Thread(
            target=self._send_statue,
            args=(payload.get("statue", {}),),
            daemon=True,
        )
        statue_thread.start()

    def stop(self):
        print("[UEBridge] Stopped")

    def _send_fountain(self, fountain: dict):
        if not fountain:
            return

        vel_min = fountain.get("VelocitySpeedMin", 200.0)
        vel_max = fountain.get("VelocitySpeedMax", 230.0)

        # Blueprint 핀 이름 내부 변환 (공백 제거)
        # Blueprint 함수에도 Force X, Force Y, Force Z 입력 핀을 만들어주세요!
        body = {
            "objectPath": ACTOR_PATH,
            "functionName": "SetFountainVariable",
            "parameters": {
                "MinSpeed": vel_min,
                "MaxSpeed": vel_max
            },
            "generateTransaction": True,
        }

        try:
            resp = requests.put(CALL_URL, json=body, timeout=0.5)
            if resp.status_code == 200:
                print(f"[UEBridge] OK | Min={vel_min} Max={vel_max}")
            else:
                print(f"[UEBridge] ERR {resp.status_code}: {resp.text[:100]}")
        except requests.exceptions.Timeout:
            print("[UEBridge] Timeout")
        except requests.exceptions.ConnectionError:
            print("[UEBridge] Connection error - is Unreal running?")

    def _send_statue(self, statue: dict):
        if not statue:
            return

        decay_amount = statue.get("DecayAmount", 0.0)

        # Statue Blueprint 또는 Material에 파라미터 쏘기 (SetScalarParameterValue 등)
        # 만약 NewBlueprint_C_1 내부에 SetStatueDecay 등의 Blueprint 함수가 있다면 그것을 호출한다고 가정합니다.
        body = {
            "objectPath": STATUE_ACTOR_PATH,
            "functionName": "SetStatueDecay",  # 블루프린트에 이 함수를 생성해주세요 (입력핀: DecayAmount)
            "parameters": {
                "DecayAmount": decay_amount,
            },
            "generateTransaction": True,
        }

        try:
            resp = requests.put(CALL_URL, json=body, timeout=0.5)
            if resp.status_code == 200:
                print(f"[UEBridge] OK (Statue) | DecayAmount={decay_amount}")
            else:
                print(f"[UEBridge] ERR (Statue) {resp.status_code}: {resp.text[:100]}")
        except requests.exceptions.Timeout:
            print("[UEBridge] Timeout (Statue)")
        except requests.exceptions.ConnectionError:
            print("[UEBridge] Connection error (Statue) - is Unreal running?")
