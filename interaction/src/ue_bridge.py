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
        thread = threading.Thread(
            target=self._send_fountain,
            args=(payload.get("fountain", {}),),
            daemon=True,
        )
        thread.start()

    def stop(self):
        print("[UEBridge] Stopped")

    def _send_fountain(self, fountain: dict):
        if not fountain:
            return

        vel_min = fountain.get("VelocitySpeedMin", 200.0)
        vel_max = fountain.get("VelocitySpeedMax", 230.0)
        
        # 새로 추가된 Linear Force XYZ 가져오기
        force_x = fountain.get("LinearForceX", 0.0)
        force_y = fountain.get("LinearForceY", 0.0)
        force_z = fountain.get("LinearForceZ", 0.0)

        # Blueprint 핀 이름 내부 변환 (공백 제거)
        # Blueprint 함수에도 Force X, Force Y, Force Z 입력 핀을 만들어주세요!
        body = {
            "objectPath": ACTOR_PATH,
            "functionName": "SetFountainVelocity",
            "parameters": {
                "MinSpeed": vel_min,
                "MaxSpeed": vel_max,
                "ForceX": force_x,
                "ForceY": force_y,
                "ForceZ": force_z,
            },
            "generateTransaction": True,
        }

        try:
            resp = requests.put(CALL_URL, json=body, timeout=0.5)
            if resp.status_code == 200:
                print(f"[UEBridge] OK | Min={vel_min} Max={vel_max} | ForceZ={force_z}")
            else:
                print(f"[UEBridge] ERR {resp.status_code}: {resp.text[:100]}")
        except requests.exceptions.Timeout:
            print("[UEBridge] Timeout")
        except requests.exceptions.ConnectionError:
            print("[UEBridge] Connection error - is Unreal running?")
