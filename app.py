import sys
import mss
import time
import os
import base64
import requests
import json
import datetime
import shutil
from collections import defaultdict, deque
from PyQt6.QtWidgets import QDialog, QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QSpinBox, QLineEdit
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QRectF
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QPainterPath, QPen
from PIL import Image
from dotenv import load_dotenv
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai

# Load environment variables
load_dotenv()
    
class ScreenCaptureThread(QThread):
    captured = pyqtSignal(QImage, str)

    def __init__(self, interval=30, max_cache_size=20):
        super().__init__()
        self.interval = interval
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.image_queue = deque(maxlen=max_cache_size)
        self.running = True
        self.count = 0

    def run(self):
        with mss.mss() as sct:
            while self.running:
                monitor = sct.monitors[0]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                # Generate a unique image ID
                image_id = f"capture_{self.count % 20}"
                image_path = os.path.join(self.debug_dir, f"{image_id}.png")

                # Save the image to the temporary directory
                img.save(image_path, format="PNG")
                
                self.count += 1
                self.image_queue.append(image_path)

                while len(self.image_queue) > self.max_cache_size:
                    old_image_path = self.image_queue.popleft()
                    os.remove(old_image_path)

                qimage = QImage(img.tobytes(), img.width, img.height, QImage.Format.Format_RGB888)
                self.captured.emit(qimage, image_id)

                for _ in range(int(self.interval * 10)):  # Check every 100ms if we should stop
                    if not self.running:
                        return
                    time.sleep(0.1)

    def stop(self):
        self.running = False
        self.wait(5000)

        # shutil.rmtree(self.debug_dir)

    def get_image_path(self, image_id):
        for image_path in self.image_queue:
            if image_id in image_path:
                return image_path
        return None
        
class DistractionAnalyzer(QThread):
    analysis_complete = pyqtSignal(bool, str)  # is_distracted, activity

    def __init__(self, possible_activities=None, blacklisted_words=None, model="llava"):
        super().__init__()
        self.possible_activities = possible_activities or []
        self.blacklisted_words = blacklisted_words or []
        self.image_path = None
        self.image_id = None
        self.model = model
        self.setup_model()

    def setup_model(self):
        if self.model == "gemini":
            genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
            self.genai_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        elif self.model == "llava":
            pass
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def set_image(self, image_path):
        self.image_path = image_path

    def set_image_id(self, image_id):
        self.image_id = image_id

    def run(self):
        if self.image_path is not None:
            options = ", ".join(self.possible_activities)
            try:
                if self.model == "gemini":
                    question = f"Describe what this person in this image is doing briefly (5 words max) from these options: {options}"
                    answer = self.ask_gemini(question, self.image_path)
                    is_distracted = self.check_distraction(answer)
                    print(f"{self.model.upper()} response: {answer}")
                elif self.model == "llava":
                    question = f"Describe what this person in this image is doing briefly (5 words max) from these options: {options}"
                    answer = self.ask_llava(question, self.image_path)
                    print(f"{self.model.upper()} response: {answer}")
                    is_distracted = self.check_distraction(answer)
                else:
                    raise ValueError(f"Unsupported model: {self.model}")
                
                self.analysis_complete.emit(is_distracted, answer.strip())
                self.image_path = None
            except Exception as e:
                print(f"Error in {self.model.upper()} analysis: {e}")
                self.analysis_complete.emit(False, "unknown")

    def check_distraction(self, activity):
        activity = activity.lower()
        for blacklisted in self.blacklisted_words:
            if blacklisted.lower() in activity:
                return True
        return False

    def ask_llava(self, prompt, image_path):
        base64_image = self.encode_image(image_path)
        
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'llava',
                'prompt': prompt,
                'images': [base64_image],
                'stream': False,
                'options': {
                    'temperature': 0
                }
            })
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}, {response.text}"

    def ask_gemini(self, prompt, image_path):
        image = Image.open(image_path)
        response = self.genai_model.generate_content([prompt, image])
        return response.text

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

class AudioThread(QThread):
    def __init__(self, text="", audio_path="Radar.mp3"):
        super().__init__()
        self.text = text
        self.audio_path = audio_path

    def run(self):
        if self.text != "":
            # tts = gTTS(text=self.text, lang='en')
            # tts.save("distraction_alert.mp3")
            playsound("distraction_alert.mp3")
            # os.remove("distraction_alert.mp3")  # Clean up the audio file
        else:
            playsound(self.audio_path)

class StatsTracker:
    def __init__(self, filename='distraction_stats.json'):
        self.filename = filename
        self.stats = self.load_stats()
        self.prev_update = None

    def load_stats(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f, object_hook=lambda d: {k: defaultdict(int, v) if k == 'activities' else v for k, v in d.items()})
        return defaultdict(lambda: {
            'distractions': 0,
            'checks': 0,
            'activities': defaultdict(int),
            'total_active_time': 0
        })

    def save_stats(self):
        with open(self.filename, 'w') as f:
            json.dump(self.stats, f, indent=2, default=lambda x: dict(x) if isinstance(x, defaultdict) else x)

    def update_stats(self, activity, is_distracted, interval):
        now = datetime.datetime.now()
        date_key = now.strftime('%Y-%m-%d')
        hour_key = now.strftime('%Y-%m-%d %H:00')

        for key in [date_key, hour_key]: # Maybe save daily and hourly stats in different files
            self.stats[key]['checks'] += 1
            if is_distracted:
                self.stats[key]['distractions'] += 1
            self.stats[key]['activities'][activity] += 1

            if self.prev_update:
                duration = (now - self.prev_update).total_seconds()
                self.stats[key]['total_active_time'] += duration

        self.prev_update = now
        self.save_stats()

    def get_stats(self, key):
        return dict(self.stats[key])
    
class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            print("Configuration file not found. Using default settings.")
            return {
                "capture_interval": 30,
                "possible_activities": ["being productive", "coding", "writing", "learning", "social media", "gaming", "watching livestream"],
                "blacklisted_words": ["social media", "gaming", "stream"],
                "notification_sound": "Radar.mp3",
            }

    def save_config(self):
        with open(self.config_file, 'w') as config_file:
            json.dump(self.config, config_file, indent=4)

class DistractionHandler:
    def __init__(self, config_manager, capture_thread=None):
        self.config_manager = config_manager
        self.distraction_popup = None
        self.audio_thread = None
        self.audio_thread2 = None
        self.capture_thread = capture_thread

    def handle_distraction(self):
        message = "You seem distracted! Get back to work!"
        if not self.distraction_popup or not self.distraction_popup.isVisible():
            self.show_distraction_popup(message)
        self.play_audio_alert(message)

    def show_distraction_popup(self, message):
        self.distraction_popup = DistractionPopup(message)
        self.distraction_popup.show()

    def play_audio_alert(self, message):
        self.audio_thread = AudioThread(text=message)
        self.audio_thread2 = AudioThread(audio_path=self.config_manager.config['notification_sound'])
        self.audio_thread.start()
        self.audio_thread2.start()

class DistractionPopup(QDialog):
    confirmed = pyqtSignal()

    def __init__(self, message, parent=None):
        super().__init__(parent, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(False)
        self.resize(400, 250)

        self.setStyleSheet("""
            QLabel, QPushButton {
                color: #FFF5E6;
                font-size: 20px;
                font-family: 'Helvetica Neue', sans-serif;
                font-weight: 300;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 30);
            }
        """)

        layout = QVBoxLayout()

        self.messageLabel = QLabel(message)
        self.messageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.messageLabel.setWordWrap(True)
        layout.addWidget(self.messageLabel)

        self.refocus_button = QPushButton("Refocus")
        self.refocus_button.clicked.connect(self.on_refocus)
        layout.addWidget(self.refocus_button)

        self.setLayout(layout)
        self.center_on_screen()

    def center_on_screen(self):
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 3
        self.move(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 10, 10)
        painter.setClipPath(path)
        painter.fillPath(path, QColor(255, 103, 0, 200))
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.drawPath(path)

    def on_refocus(self):
        self.confirmed.emit()
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Distraction Monitor")
        self.setGeometry(100, 100, 600, 250)

        self.config_manager = ConfigManager()
        self.distraction_handler = None
        self.stats_tracker = StatsTracker()

        self.setup_ui()
        self.capture_thread = None
        self.analyzer = None

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Interval setting
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Capture Interval (seconds):")
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(5, 3600)
        self.interval_spinbox.setValue(self.config_manager.config['capture_interval'])
        self.interval_spinbox.valueChanged.connect(self.save_config)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spinbox)
        layout.addLayout(interval_layout)

        # Possible activities input
        possible_layout = QHBoxLayout()
        possible_label = QLabel("Possible Activities:")
        self.possible_input = QLineEdit()
        self.possible_input.setText(", ".join(self.config_manager.config['possible_activities']))
        self.possible_input.textChanged.connect(self.save_config)
        possible_layout.addWidget(possible_label)
        possible_layout.addWidget(self.possible_input)
        layout.addLayout(possible_layout)

        # Blacklisted activities input
        blacklisted_layout = QHBoxLayout()
        blacklisted_label = QLabel("Blacklisted Activities:")
        self.blacklisted_input = QLineEdit()
        self.blacklisted_input.setText(", ".join(self.config_manager.config['blacklisted_words']))
        self.blacklisted_input.textChanged.connect(self.save_config)
        blacklisted_layout.addWidget(blacklisted_label)
        blacklisted_layout.addWidget(self.blacklisted_input)
        layout.addLayout(blacklisted_layout)

        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.toggle_monitoring)
        layout.addWidget(self.start_button)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Separate status labels
        self.monitoring_status_label = QLabel("Status: Not monitoring")
        layout.addWidget(self.monitoring_status_label)

    def toggle_monitoring(self):
        if self.start_button.text() == "Start Monitoring":
            possible_activities = [a.strip() for a in self.possible_input.text().split(',') if a.strip()]
            blacklisted_words = [a.strip() for a in self.blacklisted_input.text().split(',') if a.strip()]

            if not possible_activities:
                possible_activities = self.config_manager.config['possible_activities']
                self.possible_input.setText(", ".join(possible_activities))
            
            if not blacklisted_words:
                blacklisted_words = self.config_manager.config['blacklisted_words']
                self.blacklisted_input.setText(", ".join(blacklisted_words))

            interval = self.interval_spinbox.value()

            chosen_model = "llava"  # llava or "gemini"

            self.analyzer = DistractionAnalyzer(possible_activities, blacklisted_words, model=chosen_model)
            self.analyzer.analysis_complete.connect(self.handle_analysis_result)

            self.capture_thread = ScreenCaptureThread(interval)
            self.capture_thread.captured.connect(self.process_capture)
            self.capture_thread.start()

            self.distraction_handler = DistractionHandler(self.config_manager, self.capture_thread)

            self.start_button.setText("Stop Monitoring")
            self.monitoring_status_label.setText(f"Status: Monitoring (Interval: {interval}s)")
            self.interval_spinbox.setEnabled(False)
            self.possible_input.setEnabled(False)
            self.blacklisted_input.setEnabled(False)
        else:
            self.start_button.setEnabled(False)
            self.monitoring_status_label.setText("Status: Stopping monitoring...")
            QTimer.singleShot(100, self.stop_monitoring)

    def stop_monitoring(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None

        if self.analyzer and self.analyzer.isRunning():
            self.analyzer.quit()
            self.analyzer.wait()

        self.start_button.setText("Start Monitoring")
        self.monitoring_status_label.setText("Status: Not monitoring")
        self.interval_spinbox.setEnabled(True)
        self.possible_input.setEnabled(True)
        self.blacklisted_input.setEnabled(True)
        self.start_button.setEnabled(True)

    def process_capture(self, qimage, image_id):
        scaled_pixmap = QPixmap.fromImage(qimage).scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        if self.capture_thread:
            image_path = self.capture_thread.get_image_path(image_id)
            self.analyzer.set_image(image_path)
        else:
            print(f"Error: Could not find image with id {image_id}")

        self.analyzer.set_image_id(image_id)
        if not self.analyzer.isRunning():
            self.analyzer.start()

    def handle_analysis_result(self, is_distracted, activity):
        interval = self.config_manager.config['capture_interval']
        self.stats_tracker.update_stats(activity, is_distracted, interval)
        if is_distracted:
            self.distraction_handler.handle_distraction()

    def save_config(self):
        self.config_manager.config['capture_interval'] = self.interval_spinbox.value()
        self.config_manager.config['possible_activities'] = [a.strip() for a in self.possible_input.text().split(',') if a.strip()]
        self.config_manager.config['blacklisted_words'] = [a.strip() for a in self.blacklisted_input.text().split(',') if a.strip()]
        
        with open('config.json', 'w') as config_file:
            json.dump(self.config_manager.config, config_file, indent=4)
            
    def closeEvent(self, event):
        # Stop monitoring threads
        self.stop_monitoring()

        if self.distraction_handler:
            if self.distraction_handler.distraction_popup:
                self.distraction_handler.distraction_popup.close()
                self.distraction_handler.distraction_popup.deleteLater()

        # Clean up any remaining QTimers
        for child in self.findChildren(QTimer):
            child.stop()

        # Ensure all threads are stopped and deleted
        for child in self.findChildren(QThread):
            if child.isRunning():
                child.quit()
                child.wait()
            child.deleteLater()

        super().closeEvent(event)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    