import cv2


class WebcamCapture:
    """웹캠에서 프레임을 캡처하는 클래스."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """웹캠을 열고 해상도를 설정합니다."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {self.camera_index}번을 열 수 없습니다.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[Capture] 카메라 {self.camera_index}번 열기 성공 ({self.width}x{self.height})")

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
