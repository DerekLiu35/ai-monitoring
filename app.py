import sys
import mss
import time
import random
import os
import base64
import requests
import json
import datetime
import shutil
from collections import defaultdict, deque
from PyQt6.QtWidgets import QDialog, QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QSpinBox, QLineEdit, QSystemTrayIcon, QMenu, QComboBox
from PyQt6.QtCore import QThread, QObject, pyqtSignal, Qt, QTimer, QRectF, QPointF, QPropertyAnimation, QSize, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QIcon, QColor, QPainter, QPainterPath, QPen, QRadialGradient, QBrush
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
    analysis_complete = pyqtSignal(bool, str, str)  # is_distracted, activity, image_id

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
                    # question = f"""
                    # First, describe what this person in this image is doing briefly (e.g {options}). 
                    # Then, determine if they are doing something related to these words {", ".join(self.blacklisted_words)}.
                    # Respond with YES or NO."""
                    question = f"Describe what this person in this image is doing briefly (5 words max) from these options: {options}"
                    answer = self.ask_llava(question, self.image_path)
                    is_distracted = self.check_distraction(answer)
                    # answer = "unknown"
                    # is_distracted = True
                    print(f"{self.model.upper()} response: {answer}")
                elif self.model == "llava":
                    question = f"Describe what this person in this image is doing briefly (5 words max) from these options: {options}"
                    answer = self.ask_llava(question, self.image_path)
                    print(f"{self.model.upper()} response: {answer}")
                    is_distracted = self.check_distraction(answer)
                else:
                    raise ValueError(f"Unsupported model: {self.model}")
                
                self.analysis_complete.emit(is_distracted, answer.strip(), self.image_id)
                self.image_path = None
            except Exception as e:
                print(f"Error in {self.model.upper()} analysis: {e}")
                self.analysis_complete.emit(False, "unknown", self.image_id)

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
                "positive_reinforcement_interval": 1800,
                "positive_reinforcement_chance": 0.3
            }

    def save_config(self):
        with open(self.config_file, 'w') as config_file:
            json.dump(self.config, config_file, indent=4)

class DistractionHandler:
    def __init__(self, config_manager, capture_thread=None):
        self.config_manager = config_manager
        self.last_distraction_time = datetime.datetime.now()
        self.last_praise_time = None
        self.last_praise_time2 = datetime.datetime.now()
        self.distraction_popup = None
        self.reflection_dialog = None
        self.audio_thread = None
        self.audio_thread2 = None
        self.focus_audio_thread = None
        self.capture_thread = capture_thread
        self.misclassification_dir = "misclassifications"

    def handle_distraction(self, activity, image_id):
        self.last_distraction_time = datetime.datetime.now()
        message = "You seem distracted! Get back to work!"
        if not self.distraction_popup or not self.distraction_popup.isVisible() and not self.reflection_dialog:
            self.show_distraction_popup(message, activity, image_id)
        self.play_audio_alert(message)


    def handle_focus(self):
        current_time = datetime.datetime.now()
        if (current_time - self.last_distraction_time).total_seconds() > self.config_manager.config['positive_reinforcement_interval']:
            if not self.last_praise_time or (current_time - self.last_praise_time).total_seconds() > self.config_manager.config['positive_reinforcement_interval']:
                if current_time.hour >= 20:
                    if not self.last_praise_time2 or (current_time - self.last_praise_time2).total_seconds() > 1800:
                        if random.random() < 1/8:
                            self.focus_audio_thread = AudioThread(audio_path="bladerunner.m4a")
                            self.focus_audio_thread.start()
                            self.last_praise_time2 = current_time

    def show_distraction_popup(self, message, detected_activity, image_id):
        non_blacklisted_activities = [
            a for a in self.config_manager.config['possible_activities']
            if all(blacklisted_word not in a for blacklisted_word in self.config_manager.config['blacklisted_words'])
        ]
        self.distraction_popup = DistractionPopup(message, non_blacklisted_activities)
        self.distraction_popup.not_distracted.connect(lambda: self.handle_not_distracted(detected_activity, image_id))
        self.distraction_popup.activity_selected.connect(lambda activity: self.handle_activity_correction(detected_activity, activity, image_id))
        self.distraction_popup.confirmed.connect(self.show_reflection_dialog)
        self.distraction_popup.show()

    def handle_not_distracted(self, detected_activity, image_id):
        self.save_misclassification(detected_activity, "", image_id)
        self.hide_distraction_popup()

    def handle_activity_correction(self, detected_activity, correct_activity, image_id):
        self.save_misclassification(detected_activity, correct_activity, image_id)
        self.hide_distraction_popup()

    def save_misclassification(self, detected_activity, correct_activity, image_id):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_image_path = self.capture_thread.get_image_path(image_id)
        
        if original_image_path is None:
            print(f"Error: Could not find image with id {image_id}")
            return

        misclassification_image_path = os.path.join(self.misclassification_dir, f"misclassification_{timestamp}.png")
        misclassification_csv_path = os.path.join(self.misclassification_dir, "misclassifications.csv")

        # Copy the image
        os.makedirs(self.misclassification_dir, exist_ok=True)
        shutil.copy(original_image_path, misclassification_image_path)

        # Prepare the metadata
        metadata = {
            "timestamp": timestamp,
            "image_filename": f"misclassification_{timestamp}.png",
            "input_prompt": f"Describe what this person in this image is doing briefly (5 words max) from these options: {', '.join(self.config_manager.config['possible_activities'])}",
            "model_output": detected_activity,
            "user_correction": correct_activity if correct_activity else ""
        }

        # Check if the CSV file exists
        file_exists = os.path.isfile(misclassification_csv_path)

        # Append the metadata to the CSV file
        with open(misclassification_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metadata.keys())
            
            # Write header if file is newly created
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metadata)

        print(f"Misclassification saved: {misclassification_image_path}")
        print(f"Metadata appended to: {misclassification_csv_path}")

    def hide_distraction_popup(self):
        if self.distraction_popup:
            self.distraction_popup.hide()
            self.distraction_popup.deleteLater()
            self.distraction_popup = None

    def show_reflection_dialog(self):
        self.reflection_dialog = ReflectionDialog()
        self.reflection_dialog.refocus_clicked.connect(self.hide_reflection_dialog)
        self.reflection_dialog.show()

    def hide_reflection_dialog(self):
        if self.reflection_dialog:
            self.reflection_dialog.hide()
            self.reflection_dialog.deleteLater()
            self.reflection_dialog = None

    def play_audio_alert(self, message):
        self.audio_thread = AudioThread(text=message)
        self.audio_thread2 = AudioThread(audio_path=self.config_manager.config['notification_sound'])
        self.audio_thread.start()
        self.audio_thread2.start()

class DistractionPopup(QDialog):
    not_distracted = pyqtSignal()
    activity_selected = pyqtSignal(str)
    confirmed = pyqtSignal()

    def __init__(self, message, possible_activities, parent=None):
        super().__init__(parent, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(False)
        self.resize(400, 250)
        self.possible_activities = possible_activities

        # Custom style sheet for enhanced visuals
        self.setStyleSheet("""
            QLabel, QPushButton, QComboBox, QComboBox QAbstractItemView {
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
            QComboBox {
                background-color: rgba(255, 255, 255, 30);
                padding: 5px;
                border: none;
                border-radius: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(255, 103, 0, 200);
            }
        """)

        # Layout setup
        layout = QVBoxLayout()
        # layout.setSpacing(1)  # Reduce spacing between widgets

        self.messageLabel = QLabel(message)
        self.messageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.messageLabel.setWordWrap(True)
        layout.addWidget(self.messageLabel)

        self.refocus_button = QPushButton("Refocus")
        self.refocus_button.clicked.connect(self.on_refocus)
        layout.addWidget(self.refocus_button)

        not_distracted_layout = QHBoxLayout()
        not_distracted_layout.addStretch()
        
        self.not_distracted_button = QPushButton("Not a distraction")
        self.not_distracted_button.clicked.connect(self.on_not_distracted)
        self.not_distracted_button.setStyleSheet("""
            background-color: rgba(255, 255, 255, 10);
            color: rgba(255, 245, 230, 0.5);
            font-size: 12px;
            padding: 4px;
            max-width: 120px;
        """)
        not_distracted_layout.addWidget(self.not_distracted_button)

        layout.addLayout(not_distracted_layout)

        self.setLayout(layout)
        self.center_on_screen()

        # Setup breathing effect for opacity
        # self.breathing_animation = QPropertyAnimation(self, b"windowOpacity")
        # self.breathing_animation.setDuration(4000)  # Duration for fade in/out
        # self.breathing_animation.setStartValue(0.7)
        # self.breathing_animation.setEndValue(1.0)
        # self.breathing_animation.setLoopCount(-1)  # Loop indefinitely
        # self.breathing_animation.setEasingCurve(QEasingCurve.Type.SineCurve)

        # self.size_animation = QPropertyAnimation(self, b"geometry")
        # self.size_animation.setDuration(4000)  # 4 seconds
        # self.size_animation.setStartValue(self.geometry())
        # self.size_animation.setEndValue(self.geometry().adjusted(-20, -20, 20, 20))  # Slightly grow and shrink
        # self.size_animation.setLoopCount(-1)
        # self.size_animation.setEasingCurve(QEasingCurve.Type.SineCurve)

        # self.breathing_animation.start()
        # self.size_animation.start()

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
        painter.fillPath(path, QColor(255, 103, 0, 200))  # Soft orange color
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.drawPath(path)

    def on_refocus(self):
        self.confirmed.emit()
        self.accept()

    def on_not_distracted(self):
        correction_dialog = QDialog(self, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        correction_dialog.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        correction_dialog.setModal(True)
        correction_dialog.resize(300, 200)
        correction_dialog.setStyleSheet(self.styleSheet())

        main_layout = QVBoxLayout(correction_dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel("Select the correct activity (optional):")
        label.setStyleSheet("color: white; font-size: 14px;")
        main_layout.addWidget(label)

        combo = QComboBox()
        combo.addItem("Select an activity")
        combo.addItems(self.possible_activities)
        main_layout.addWidget(combo)

        button_box = QHBoxLayout()
        skip_button = QPushButton("Skip")
        skip_button.clicked.connect(correction_dialog.reject)
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(correction_dialog.accept)

        button_box.addWidget(skip_button)
        button_box.addWidget(confirm_button)
        main_layout.addLayout(button_box)

        # Custom paint event for the correction dialog
        def paintEvent(event):
            painter = QPainter(correction_dialog)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            path = QPainterPath()
            path.addRoundedRect(QRectF(correction_dialog.rect()), 10, 10)
            painter.setClipPath(path)
            painter.fillPath(path, QColor(255, 103, 0, 200))  # Soft orange color
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
            painter.drawPath(path)

        correction_dialog.paintEvent = paintEvent

        result = correction_dialog.exec()

        if result == QDialog.DialogCode.Accepted and combo.currentIndex() != 0:
            self.activity_selected.emit(combo.currentText())
        else:
            self.not_distracted.emit()

        self.accept()

class ReflectionDialog(QDialog):
    refocus_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(False)
        self.resize(400, 250)
        self.setStyleSheet("""
            QLabel, QLineEdit, QPushButton {
                color: #FFF5E6;
                font-size: 20px;
                font-family: 'Helvetica Neue', sans-serif;
                font-weight: 300;
            }
            QLineEdit {
                background-color: rgba(255, 255, 255, 20);
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 50);
                padding: 8px;
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
        
        reflection_label = QLabel("What caused the distraction?")
        reflection_label.setWordWrap(True)
        reflection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(reflection_label)

        self.reflection_input = QLineEdit()
        self.reflection_input.setPlaceholderText("Reflect on your distraction...")
        layout.addWidget(self.reflection_input)

        confirm_button = QPushButton("Refocus")
        confirm_button.clicked.connect(self.on_refocus_clicked)
        layout.addWidget(confirm_button, alignment=Qt.AlignmentFlag.AlignCenter)

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
        painter.fillPath(path, QColor(230, 90, 90, 200))  # Soft red color
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.drawPath(path)

    def on_refocus_clicked(self):
        self.refocus_clicked.emit()
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

    def handle_analysis_result(self, is_distracted, activity, image_id):
        interval = self.config_manager.config['capture_interval']
        self.stats_tracker.update_stats(activity, is_distracted, interval)
        if is_distracted:
            self.distraction_handler.handle_distraction(activity, image_id)
        else:
            self.distraction_handler.handle_focus()

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
            if self.distraction_handler.reflection_dialog:
                self.distraction_handler.reflection_dialog.close()
                self.distraction_handler.reflection_dialog.deleteLater()

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
    