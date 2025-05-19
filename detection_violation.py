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

# -----------------------------------------------------------------------------------------------
# Variable
# -----------------------------------------------------------------------------------------------
stream_url = "rtmp://a.rtmp.youtube.com/live2/wdr2-9vr0-18ha-gfvm-c60x"
last_update_data = {
    "bike_count": -1,
    "violation_count": -1
}
bike_count_doc_ref = None 
# -----------------------------------------------------------------------------------------------

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
        self.annotated_frame = None
        
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
                image_bytes, metadata = self.violation_queue.get()
                self._upload_violation(image_bytes, metadata)
                self.violation_queue.task_done()
            except Exception as e:
                print(f"[THREAD ERROR] {e}")
                
    def _upload_violation(self, image_bytes, metadata):
        print(f"[DEBUG] Uploading violation: {metadata['filename']}")
        try:
            blob = bucket.blob(f"pelanggaran/{metadata['filename']}")
            blob.upload_from_string(image_bytes, content_type='image/jpeg')
        except Exception as e:
            print(f"[UPLOAD ERROR] {e}")
        blob.make_public()

        doc_id = f"pelanggaran_{metadata['tanggal']}_{metadata['waktu']}_{metadata['track_id']}"
        db.collection("pelanggaran").document(doc_id).set({
            "pelanggaran": metadata['filename'],
            "tanggal": metadata['tanggal'],
            "waktu": metadata['waktu'],
            "url": blob.public_url
        })
        print(f"[UPLOAD] Pelanggaran disimpan: {metadata['filename']}")

# -----------------------------------------------------------------------------------------------
# Suport Function function
# -----------------------------------------------------------------------------------------------
def _init_ffmpeg_if_needed(user_data, frame, width, height):
    if user_data.ffmpeg_process is None and frame is not None:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', '30', '-i', '-',
            '-f', 'lavfi', '-i', 'anullsrc',
            '-c:v', 'libx264', '-preset', 'veryfast',
            '-b:v', '3000k', '-maxrate', '3000k', '-bufsize', '6000k',
            '-c:a', 'aac', '-b:a', '128k',
            '-f', 'flv', stream_url
        ]
        user_data.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        print("[FFMPEG] Streaming process started.")

def _send_frame_to_ffmpeg(user_data, frame):
    if frame is not None and user_data.ffmpeg_process is not None:
        try:
            user_data.ffmpeg_process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"[FFMPEG ERROR] {e}")

def _init_line_zone(user_data, width, height):
    x_center = width // 2
    user_data.line_zone = sv.LineZone(
        start=sv.Point(x_center, height),
        end=sv.Point(x_center, 0)
    )
    user_data.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

def _classify_detections(detections, width, height):
    bikes, helmets, unhelmets = [], [], []
    for det in detections:
        label = det.get_label()
        bbox = det.get_bbox()
        x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
        x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if label == "bike":
            track = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = track[0].get_id() if track else None
            bikes.append({"bbox": [x1, y1, x2, y2], "conf": det.get_confidence(), "track_id": track_id, "helmet": [], "unhelmet": []})
        elif label == "helmet":
            helmets.append({"center": center})
        elif label == "unhelmet":
            unhelmets.append({"center": center})
    return {"bikes": bikes, "helmets": helmets, "unhelmets": unhelmets}

def _associate_helmet_status(detections):
    for bike in detections["bikes"]:
        x1, y1, x2, y2 = bike["bbox"]
        for h in detections["helmets"]:
            cx, cy = h["center"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                bike["helmet"].append(h)
        for uh in detections["unhelmets"]:
            cx, cy = uh["center"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                bike["unhelmet"].append(uh)

def _prepare_sv_detections(bikes):
    if not bikes["bikes"]:
        return None, None

    xyxy, confs, class_ids, track_ids, violations = [], [], [], [], []

    for bike in bikes["bikes"]:
        xyxy.append(bike["bbox"])
        confs.append(bike["conf"])
        class_ids.append(0)
        track_ids.append(bike["track_id"])
        violations.append(1 if len(bike["unhelmet"]) > 0 else 0)

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=int),
        confidence=np.array(confs),
        class_id=np.array(class_ids),
        tracker_id=np.array(track_ids)
    )
    return detections, violations

def _update_crossed_ids(user_data, detections):
    before = user_data.line_zone.in_count
    user_data.line_zone.trigger(detections=detections)
    after = user_data.line_zone.in_count

    if after > before:
        for i in range(len(detections)):
            tid = detections.tracker_id[i]
            if tid is not None:
                user_data.crossed_track_ids.add(tid)

def _handle_violations(user_data, detections, violations, frame):
    for i, is_violation in enumerate(violations):
        track_id = detections.tracker_id[i]
        if track_id in user_data.crossed_track_ids and is_violation and track_id not in user_data.counted_violation_ids:
            user_data.violation_count += 1
            user_data.counted_violation_ids.add(track_id)
            
            if frame is not None:
                x1, y1, x2, y2 = detections.xyxy[i]

                # Validasi agar bbox tidak keluar dari dimensi frame
                height, width, _ = frame.shape
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    success, buffer = cv2.imencode('.jpg', crop_rgb)
                    if success:
                        now = datetime.datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time_str = now.strftime("%H-%M-%S")
                        filename = f"violation_{date}_{time_str}_{track_id}.jpg"
                        metadata = {
                            "tanggal": date,
                            "waktu": time_str,
                            "track_id": track_id,
                            "filename": filename
                        }
                        user_data.violation_queue.put((buffer.tobytes(), metadata))
                    else:
                        print(f"[ERROR] Gagal encode image untuk track_id {track_id}")
                else:
                    print(f"[WARNING] Bounding box tidak valid untuk track_id {track_id}")

def _log_counts(user_data):
    print(f"[INFO] Frame {user_data.get_count()} - Bike Count (Right-to-Left): {user_data.line_zone.in_count}")
    print(f"[INFO] Jumlah Pelanggaran: {user_data.violation_count}")

def _annotate_frame(user_data, frame):
    annotated = frame.copy()
    user_data.line_annotator.annotate(
        frame=annotated,
        line_counter=user_data.line_zone
    )
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    #user_data.set_frame(annotated_bgr)
    user_data.annotated_frame = annotated_bgr
    # print(f'[YTYT] frame yt (_annotate_frame)= {user_data.annotated_frame}')



# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    format, width, height = get_caps_from_pad(pad)
    user_data.frame_width = width
    user_data.frame_height = height
    
    user_data.use_frame =True

    frame = None
    if format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    _init_ffmpeg_if_needed(user_data, frame, width, height)

    if user_data.line_zone is None:
        _init_line_zone(user_data, width, height)

    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
    bike_detections = _classify_detections(detections, width, height)

    _associate_helmet_status(bike_detections)
    
    sv_detections, violation_status = _prepare_sv_detections(bike_detections)

    if sv_detections is not None:
        _update_crossed_ids(user_data, sv_detections)
        _handle_violations(user_data, sv_detections, violation_status, frame)
        _log_counts(user_data)
        # print(f'sv_detection {user_data.use_frame} |-| {frame}')

        if user_data.use_frame and frame is not None:
            print('sv_detection in if []')
            _annotate_frame(user_data, frame)
            
        frame_to_send = user_data.annotated_frame if user_data.annotated_frame is not None else frame
        # print(f'[YTYT] frame yt (_app_callback)= {user_data.annotated_frame}')
        _send_frame_to_ffmpeg(user_data, frame_to_send)

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
        
    bike_count = user_data.line_zone.in_count
    violation_count = user_data.violation_count

    # Hindari update jika tidak ada perubahan
    if (bike_count == last_update_data["bike_count"] and violation_count == last_update_data["violation_count"]):
        return
        
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H-%M-%S")
    end_time = f"{date}_{time}"
    
    try:
        bike_count_doc_ref.update({
            "waktu_akhir": end_time,
            "jumlah_motor": bike_count,
            "jumlah_pelanggaran": violation_count
        })
        last_update_data["bike_count"] = bike_count
        last_update_data["violation_count"] = violation_count
        print(f"[UPDATE] Firestore updated: {bike_count=} {violation_count=} {end_time=}")
    except Exception as e:
        print(f"[ERROR] Failed to update Firestore: {e}")
    
# Scheduler for Update Bikes Counting Record Function
def schedule_updates(user_data):
    def update(_):
        threading.Thread(
            target=update_bike_count_document,
            args=(user_data,),
            daemon=True
        ).start()
        return True  # terus berulang
    GLib.timeout_add_seconds(5, update, None)

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    initialize_bike_count_document()
    schedule_updates(user_data)
    
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
