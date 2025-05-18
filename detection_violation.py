import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import supervision as sv

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

import datetime
import pylivestream.api as pls
import subprocess
import threading
import queue


import firebase_admin
from firebase_admin import credentials, firestore, storage

# -----------------------------------------------------------------------------------------------
# initialitation Firebase
# -----------------------------------------------------------------------------------------------
cred = credentials.Certificate("capstonesiph-d4637-firebase-adminsdk-fbsvc-89d8089db2.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'capstonesiph-d4637.firebasestorage.app' 
})

db = firestore.client()
bucket = storage.bucket()

bike_count_doc_ref = None 
stream_url = "rtmp://a.rtmp.youtube.com/live2/wdr2-9vr0-18ha-gfvm-c60x"

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_width = None
        self.frame_height = None
        self.line_zone = None
        self.line_annotator = None
        self.crossed_track_ids = set()
        self.counted_violation_ids = set()
        self.counted_compliant_ids = set()
        self.violation_count = 0
        self.compliant_count = 0
        
        self.use_frame = True

        self.ffmpeg_process = None
        self.frame_sent = False
        
        self.violation_queue = queue.Queue()
        self.violation_saver_thread = threading.Thread(target=self._violation_saver_worker, daemon=True)
        self.violation_saver_thread.start()
        
    def _violation_saver_worker(self):
        while True:
            try:
                frame, bbox, track_id = self.violation_queue.get()
                save_violation_crop(frame, bbox, track_id, db, bucket)
                self.violation_queue.task_done()
            except Exception as e:
                print(f"[THREAD ERROR] {e}")

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    user_data.use_frame = True
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    user_data.frame_width = width
    user_data.frame_height = height
    
    # DEBUG FORMAT SIZE BUFFER USE_DATA.USE_FRAME
    #print(f"[DEBUG] Frame format: {format}, size: {width}x{height}")
    #print(f"[DEBUG] Frame user_data.use_frame {user_data.use_frame}, size: {width}x{height} | Buffer {buffer}")
    
    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    if user_data.ffmpeg_process is None and frame is not None:
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo', 
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{width}x{height}',
                '-r', '30',
                '-i', '-',
                '-f', 'lavfi', '-i', 'anullsrc',
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-b:v', '3000k',
                '-maxrate', '3000k',
                '-bufsize', '6000k',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-f', 'flv',
                stream_url
            ]
            user_data.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            print("[FFMPEG] Streaming process started.")
        
    if frame is not None and user_data.ffmpeg_process is not None:
        # Kirim frame ke ffmpeg stdin
        try:
            user_data.ffmpeg_process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"[FFMPEG ERROR] {e}")

    # DEBUG FRAME
    '''
    if frame is None:
        print(f"[ERROR] get_numpy_from_buffer returned None | Frame shape: {frame}, dtype: {frame}")
    else:
        print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
    '''
        

    if user_data.line_zone is None:
        x_center = width // 2
        user_data.line_zone = sv.LineZone(start=sv.Point(x_center, height), end=sv.Point(x_center, 0))
        user_data.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    bike_detections = []
    helmet_detections = []
    unhelmet_detections = []

    for det in detections:
        label = det.get_label()
        bbox = det.get_bbox()
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)
        conf = det.get_confidence()

        if label == "bike":
            track = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = track[0].get_id() if len(track) == 1 else None
            bike_detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "track_id": track_id,
                "helmet": [],
                "unhelmet": []
            })
        elif label == "helmet":
            helmet_detections.append({"center": ((x1 + x2) // 2, (y1 + y2) // 2)})
        elif label == "unhelmet":
            unhelmet_detections.append({"center": ((x1 + x2) // 2, (y1 + y2) // 2)})

    for bike in bike_detections:
        x1, y1, x2, y2 = bike["bbox"]
        for h in helmet_detections:
            cx, cy = h["center"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                bike["helmet"].append(h)
        for uh in unhelmet_detections:
            cx, cy = uh["center"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                bike["unhelmet"].append(uh)

    xyxy = []
    confidences = []
    class_ids = []
    tracker_ids = []
    violation_status = []

    for bike in bike_detections:
        xyxy.append(bike["bbox"])
        confidences.append(bike["conf"])
        class_ids.append(0)
        tracker_ids.append(bike["track_id"])
        is_violation = len(bike["unhelmet"]) > 0
        violation_status.append(1 if is_violation else 0)

    if len(xyxy) > 0:
        sv_detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=int),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            tracker_id=np.array(tracker_ids)
        )

        before_count = user_data.line_zone.in_count
        user_data.line_zone.trigger(detections=sv_detections)
        after_count = user_data.line_zone.in_count

        if after_count > before_count:
            for i in range(len(sv_detections)):
                tid = sv_detections.tracker_id[i]
                if tid is not None:
                    user_data.crossed_track_ids.add(tid)

        for i, v in enumerate(violation_status):
            track_id = sv_detections.tracker_id[i]
            if track_id in user_data.crossed_track_ids:
                if v == 1 and track_id not in user_data.counted_violation_ids:
                    user_data.violation_count += 1
                    user_data.counted_violation_ids.add(track_id)
                    if user_data.use_frame and format and width and height:
                        user_data.violation_queue.put((frame.copy(), sv_detections.xyxy[i], track_id))

        print(f"[INFO] Frame {user_data.get_count()} - Bike Count (Right-to-Left): {user_data.line_zone.in_count}")
        print(f"[INFO] Jumlah Pelanggaran: {user_data.violation_count}")

    if user_data.use_frame and frame is not None:
        annotated_frame = frame.copy()
        user_data.line_annotator.annotate(
            frame=annotated_frame,
            line_counter=user_data.line_zone
        )
        cv2.putText(annotated_frame, f"Right-to-Left: {user_data.line_zone.in_count}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(annotated_frame)

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Initialitation Function for Bikes Counting Record
# -----------------------------------------------------------------------------------------------
def initialize_bike_count_document():
    global bike_count_doc_ref
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H-%M-%S")
    start_time = f"{date}_{time}"
    bike_count_doc_ref = db.collection("jumlah_motor").document(f"jumlah_motor_{start_time}")
    bike_count_doc_ref.set({
        "waktu_mulai": start_time,
        "waktu_akhir": start_time,
        "jumlah_motor": 0,
        "jumlah_pelanggaran": 0,
    })
    print(f"[INIT] bike_count document created with start_time: {start_time}")
    
# -----------------------------------------------------------------------------------------------
# Update Function for Bikes Counting Record
# -----------------------------------------------------------------------------------------------
def update_bike_count_document(user_data):
    if bike_count_doc_ref is None:
        print("[ERROR] bike_count_doc_ref is not initialized.")
        return
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H-%M-%S")
    end_time = f"{date}_{time}"
    bike_count = user_data.line_zone.in_count
    violation_count = user_data.violation_count
    bike_count_doc_ref.update({
        "waktu_akhir": end_time,
        "jumlah_motor": bike_count,
        "jumlah_pelanggaran": violation_count
    })
    print(f"[UPDATE] bike_count updated: end_time={end_time}, Bikes={bike_count}, Violations={violation_count}")
    
# Scheduler for Update Bikes Counting Record Function
def schedule_updates(user_data):
    def update(_):
        update_bike_count_document(user_data)
        return True  # agar terus berulang
    GLib.timeout_add_seconds(5, update, None)

# -----------------------------------------------------------------------------------------------
# Save Violation
# -----------------------------------------------------------------------------------------------
def save_violation_crop(frame, bbox, track_id, db, bucket):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    date = datetime.datetime.now().strftime("%Y-%m-%d") 
    time = datetime.datetime.now().strftime("%H-%M-%S")
    filename = f"violation_{date}_{time}_{track_id}.jpg"
    
    '''
    # save img in locak disk
    os.makedirs("violations", exist_ok=True)
    filepath = os.path.join("violations", filename)
    cv2.imwrite(filepath, rgb_image)
    print(f"[SAVE IMG] save unhelmet violation img: {filename}")
    '''
    # save img in memory
    success, buffer = cv2.imencode('.jpg', rgb_image)
    if not success:
        print(f"[ERROR] Gagal mengencode image untuk track_id {track_id}")
        return

    # Upload ke Firebase Storage
    blob = bucket.blob(f"pelanggaran/{filename}")
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
    blob.make_public()     
    
    # Simpan metadata ke Firestore
    doc_id = f"pelanggaran_{date}_{time}_{track_id}"
    db.collection("pelanggaran").document(doc_id).set({
        "pelanggaran": filename,
        "tanggal": date,
        "waktu": time,
        "url": blob.public_url
    })

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    initialize_bike_count_document()
    schedule_updates(user_data)
    
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
