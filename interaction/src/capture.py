import cv2


class WebcamCapture:
    """웹캠에서 프레임을 캡처하는 클래스."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """웹캠을 열고 해상도 및 안정화 속성을 설정합니다."""
        # 기본 백엔드로 시도하되 버퍼를 최소화하여 지연 및 끊김 방지
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {self.camera_index}번을 열 수 없습니다.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        real_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Capture] 카메라 {self.camera_index}번 연결 성공 (실제해상도: {real_w}x{real_h})")

    def is_opened(self) -> bool:
        """카메라가 현재 열려있는지 확인합니다."""
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """
        프레임을 읽어 반환합니다.

        Returns:
            (success: bool, frame: np.ndarray | None)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        """웹캠 자원을 해제합니다."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[Capture] 카메라 자원 해제 완료")
