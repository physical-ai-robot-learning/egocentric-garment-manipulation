"""11-12 user "orbbec 각도가 좀 틀어졌어 이거 다시 맞추게 해줘": minimal Orbbec live-frame server for the camera-alignment page
(laundry_deploy.tools.cam_align, http://localhost:8772 with CAM_ALIGN_SRC=http://127.0.0.1:8092/frame) when no recorder is running.
Serves the raw colour frame (no overlays) as one JPEG per GET /frame.  Records nothing.  Stop it (Ctrl+C) before recording:
only one process can open the Orbbec.
    sudo .venv/bin/python -m laundry_shared.visual_teleop.tools.orbbec_frame_server        (sudo: the Orbbec SDK needs it on this Mac)
"""
from __future__ import annotations

import http.server
import threading

import cv2

PORT = 8092
LATEST = {"jpg": b""}


def main() -> None:
    from laundry_shared.ego_collector.camera.capture import CaptureConfig
    from laundry_shared.ego_collector.camera.orbbec_capture import OrbbecCapture
    cap = OrbbecCapture(CaptureConfig(index=0, width=848, height=480, fps=30)); cap.open()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            b = LATEST["jpg"]
            self.send_response(200 if b else 503); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass

    threading.Thread(target=http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever, daemon=True).start()
    print(f"[오비] 정렬용 영상 http://127.0.0.1:{PORT}/frame  (끝내려면 Ctrl+C, 녹화 전에 꼭 끄세요)", flush=True)
    try:
        while True:
            f = cap.get(timeout=0.5)
            if f is not None:
                LATEST["jpg"] = cv2.imencode(".jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()


if __name__ == "__main__":
    main()
