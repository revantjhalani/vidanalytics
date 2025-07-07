from fastapi import FastAPI, WebSocket, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import json
import asyncio
import time
import base64
import numpy as np
from typing import Dict, List, Set
import uuid
from collections import defaultdict
import math
import threading
import queue
from contextlib import asynccontextmanager
import os
import tempfile
import shutil

# --- GLOBAL VARIABLES ---
uploaded_videos_dir = "uploaded_videos"
current_video_path = None
processing_active = False
processing_lock = threading.Lock()

# Create uploads directory if it doesn't exist
os.makedirs(uploaded_videos_dir, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Video Analytics server starting up...")
    yield
    # Shutdown
    print("Video Analytics server shutting down...")
    cleanup_resources()

app = FastAPI(title="Video Analytics Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- MODEL INITIALIZATION ---
print("Loading YOLOv12 model...")
yolo_model = YOLO('yolo11n.pt')
print("YOLOv12 model loaded.")

# --- TRACKING AND ANALYTICS ---
class PersonTracker:
    def __init__(self):
        self.active_people: Dict[str, dict] = {}
        self.unique_people_seen: Set[str] = set()
        self.people_history: List[dict] = []
        self.frame_count = 0
        self.start_time = time.time()
        self.next_person_id = 1
        self.max_distance_threshold = 100  # Maximum distance to consider same person
        self.max_missing_frames = 30  # Remove person after 30 frames of absence
        
    def calculate_distance(self, box1, box2):
        """Calculate distance between two bounding box centers"""
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        return math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    
    def find_groups(self, people_data, group_distance_threshold=150):
        """Find groups of people based on proximity"""
        if len(people_data) < 2:
            return []
        
        groups = []
        visited = set()
        
        for i, person1 in enumerate(people_data):
            if i in visited:
                continue
                
            group = [i]
            visited.add(i)
            
            for j, person2 in enumerate(people_data):
                if j in visited or i == j:
                    continue
                
                distance = self.calculate_distance(person1['bbox'], person2['bbox'])
                if distance < group_distance_threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def match_detections_to_tracks(self, detections):
        """Match new detections to existing tracks"""
        matched_people = []
        used_track_ids = set()
        
        for detection in detections:
            best_match_id = None
            best_distance = float('inf')
            
            # Try to match with existing active people
            for person_id, person_data in self.active_people.items():
                if person_id in used_track_ids:
                    continue
                    
                distance = self.calculate_distance(detection['bbox'], person_data['bbox'])
                if distance < self.max_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = person_id
            
            if best_match_id:
                # Update existing person
                person_data = self.active_people[best_match_id]
                person_data.update({
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'attributes': detection.get('attributes', person_data.get('attributes', {})),
                    'timestamp': time.time(),
                    'missing_frames': 0
                })
                matched_people.append(person_data)
                used_track_ids.add(best_match_id)
            else:
                # Create new person
                person_id = f"P{self.next_person_id:03d}"
                self.next_person_id += 1
                self.unique_people_seen.add(person_id)
                
                person_data = {
                    'id': person_id,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'attributes': detection.get('attributes', {}),
                    'timestamp': time.time(),
                    'missing_frames': 0
                }
                self.active_people[person_id] = person_data
                matched_people.append(person_data)
        
        return matched_people
    
    def cleanup_missing_people(self):
        """Remove people who haven't been seen for too long"""
        to_remove = []
        for person_id, person_data in self.active_people.items():
            person_data['missing_frames'] += 1
            if person_data['missing_frames'] > self.max_missing_frames:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.active_people[person_id]
    
    def reset(self):
        """Reset the tracker to start fresh"""
        self.active_people.clear()
        self.unique_people_seen.clear()
        self.people_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self.next_person_id = 1
    
    def update(self, detections):
        """Update tracking with new detections"""
        self.frame_count += 1
        
        # Match detections to existing tracks
        current_people = self.match_detections_to_tracks(detections)
        
        # Clean up people who haven't been detected recently
        self.cleanup_missing_people()
        
        # Find groups among current people
        groups = self.find_groups(current_people)
        
        # Calculate average confidence
        avg_confidence = 0
        if current_people:
            total_confidence = sum(person['confidence'] for person in current_people)
            avg_confidence = total_confidence / len(current_people)
        
        # Update statistics
        stats = {
            'current_people_count': len(current_people),
            'unique_people_seen': len(self.unique_people_seen),
            'groups': len(groups),
            'group_details': groups,
            'uptime': time.time() - self.start_time,
            'total_frames': self.frame_count,
            'fps': self.frame_count / (time.time() - self.start_time) if time.time() - self.start_time > 0 else 0,
            'avg_confidence': avg_confidence
        }
        
        return current_people, stats

tracker = PersonTracker()

def cleanup_resources():
    """Clean up resources on shutdown"""
    global current_video_path, processing_active
    
    # Stop any active processing
    with processing_lock:
        processing_active = False
    
    # Clean up temporary video files
    if current_video_path and os.path.exists(current_video_path):
        try:
            os.remove(current_video_path)
        except:
            pass
    
    # Clean up uploads directory
    if os.path.exists(uploaded_videos_dir):
        try:
            shutil.rmtree(uploaded_videos_dir)
            os.makedirs(uploaded_videos_dir, exist_ok=True)
        except:
            pass

# --- VIDEO PROCESSING ---
frame_analysis_counter = 0
deepface_skip_frames = 1  # Run DeepFace analysis every 5th frame

def analyze_frame(frame):
    """Analyze a single frame and return detection data"""
    global frame_analysis_counter
    frame_analysis_counter += 1
    
    detections = []
    
    try:
        # Run YOLO detection
        yolo_results = yolo_model(frame, classes=[0], verbose=False)
        
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    if confidence < 0.3:  # Lower threshold for more consistent detection
                        continue
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'attributes': {}
                    }
                    
                    # Run DeepFace analysis less frequently for better performance
                    if frame_analysis_counter % deepface_skip_frames == 0:
                        try:
                            padding = 30
                            y1_crop = max(0, y1-padding)
                            y2_crop = min(frame.shape[0], y2+padding)
                            x1_crop = max(0, x1-padding)
                            x2_crop = min(frame.shape[1], x2+padding)
                            
                            person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                            
                            # Ensure minimum crop size and valid dimensions
                            if (person_crop.size > 0 and 
                                person_crop.shape[0] > 60 and person_crop.shape[1] > 60 and
                                len(person_crop.shape) == 3):
                                
                                # Create a copy to avoid memory issues
                                crop_copy = person_crop.copy()
                                
                                analysis_results = DeepFace.analyze(
                                    img_path=crop_copy,
                                    actions=['age', 'gender', 'emotion', 'race'],
                                    enforce_detection=False,
                                    silent=True
                                )
                                
                                if isinstance(analysis_results, list) and len(analysis_results) > 0:
                                    face_data = analysis_results[0]
                                    detection['attributes'] = {
                                        'age': round(face_data.get('age', 'Unknown') * 0.7, 1), 
                                        'gender': face_data.get('dominant_gender', 'Unknown').capitalize(),
                                        'emotion': face_data.get('dominant_emotion', 'Unknown').capitalize(),
                                        'race': face_data.get('dominant_race', 'Unknown').capitalize()
                                    }
                                
                                # Explicitly delete the crop copy
                                del crop_copy
                        
                        except Exception as e:
                            # Skip if face analysis fails
                            pass
                    
                    detections.append(detection)
                    
    except Exception as e:
        print(f"YOLO detection error: {e}")
        detections = []
    
    return detections

# --- FILE UPLOAD ---
async def save_uploaded_video(file: UploadFile) -> str:
    """Save uploaded video file and return the file path"""
    # Generate unique filename
    filename = file.filename or "video"
    file_extension = os.path.splitext(filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(uploaded_videos_dir, unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_path

# --- API ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """Handle video file upload"""
    # Validate file type
    if video.content_type and not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Validate file size (100MB limit)
    max_size = 100 * 1024 * 1024  # 100MB in bytes
    if video.size and video.size > max_size:
        raise HTTPException(status_code=400, detail="File size must be less than 100MB")
    
    try:
        # Save the uploaded file
        file_path = await save_uploaded_video(video)
        global current_video_path
        current_video_path = file_path
        
        return JSONResponse(content={
            "message": "Video uploaded successfully",
            "filename": os.path.basename(file_path)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

@app.websocket("/ws/process")
async def websocket_process_video(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    
    try:
        while True:
            # Wait for processing command
            message = await websocket.receive_text()
            print(f"Received WebSocket message: {message}")
            data = json.loads(message)
            
            if data.get('action') == 'start_processing':
                filename = data.get('filename')
                print(f"Starting processing for file: {filename}")
                print(f"Current video path: {current_video_path}")
                
                if not current_video_path or not os.path.exists(current_video_path):
                    error_msg = f'Video file not found: {current_video_path}'
                    print(error_msg)
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': error_msg
                    }))
                    continue
                
                # Process the video
                await process_video_file(websocket, current_video_path)
                break  # Exit after processing is complete
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': f'Processing error: {str(e)}'
            }))
        except:
            pass
    finally:
        print("WebSocket connection closing")
        try:
            await websocket.close()
        except:
            pass

async def process_video_file(websocket: WebSocket, video_path: str):
    """Process uploaded video file and send frames via WebSocket"""
    global processing_active
    
    print(f"Starting video processing for: {video_path}")
    
    with processing_lock:
        processing_active = True
    
    # Reset tracker for fresh start
    tracker.reset()
    
    try:
        # Open video file
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f'Could not open video file: {video_path}'
            print(error_msg)
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': error_msg
            }))
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {fps} fps, {width}x{height}")
        
        frame_count = 0
        last_progress_sent = 0
        
        while processing_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No more frames to read")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better performance
            if frame_count % 3 == 0:
                try:
                    print(f"Processing frame {frame_count}")
                    # Try to analyze frame, but continue even if AI models fail
                    try:
                        detections = analyze_frame(frame)
                        print(f"Found {len(detections)} detections")
                        
                        # Update tracker
                        people_data, stats = tracker.update(detections)
                    except Exception as ai_error:
                        print(f"AI analysis failed, continuing with empty detections: {ai_error}")
                        # Continue with empty detections if AI models fail
                        detections = []
                        people_data, stats = tracker.update(detections)
                    
                    # Encode frame to base64
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    
                    # Send frame data
                    frame_data = {
                        'type': 'frame',
                        'frame': frame_base64,
                        'people': people_data,
                        'stats': stats,
                        'frame_dimensions': {'width': frame.shape[1], 'height': frame.shape[0]}
                    }
                    
                    print(f"Sending frame {frame_count} data")
                    await websocket.send_text(json.dumps(frame_data))
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Send progress update every 10 frames or 5% progress
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            if frame_count % 10 == 0 or progress - last_progress_sent >= 5:
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                await websocket.send_text(json.dumps({
                    'type': 'progress',
                    'progress': progress,
                    'current_frame': frame_count,
                    'total_frames': total_frames
                }))
                last_progress_sent = progress
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        print("Video processing completed")
        # Send completion message
        await websocket.send_text(json.dumps({
            'type': 'complete'
        }))
        
    except Exception as e:
        print(f"Video processing error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_text(json.dumps({
            'type': 'error',
            'message': f'Processing error: {str(e)}'
        }))
    finally:
        cap.release()
        with processing_lock:
            processing_active = False
        print("Video processing cleanup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 