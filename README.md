# ai-monitoring

## Features
- Capture screen activity at custom intervals.
- Analyze screen captures for distracting activities using AI models.
- Display visual and audio alerts for detected distractions.
- Track statistics of distractions and activities.
- Customizable activities and alert settings.

## Installation & Setup

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/DerekLiu35/ai-monitoring.git
cd ai-monitoring
```

### Step 2: Dependencies Installation
1. **Python 3.7+**
2. **PIP** (Python package manager)

To keep your project dependencies organized and avoid potential conflicts with other Python projects, it's recommended to use a Python virtual environment (`venv`) (or conda). Here’s how to set up and activate a virtual environment before installing dependencies.

In the project directory, run:
```bash
python3 -m venv venv
```
This will create a new folder named `venv` containing a local Python environment.

Then,

- **On macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

- **On Windows**:
  ```bash
  .\venv\Scripts\activate
  ```

After activation, you’ll see `(venv)` at the beginning of your command prompt, indicating that the virtual environment is active.

With the virtual environment activated, install the project dependencies using:
```bash
pip3 install -r requirements.txt
```

### Step 3: Model Selection and Configuration

This application uses AI models to analyze screen captures and determine if the user is engaged in distracting activities. The current setup supports two model options:

1. **Gemini Model**:
   - **Model Name**: `gemini-1.5-flash-latest`
   - This model is available via the Google Gemini API. For more accurate analysis, Gemini is recommended.
   - **Configuration**: Add your Gemini API key to the `.env` file as follows:
     ```plaintext
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

2. **LLaVA Model via Ollama**:
   - **Ollama** provides a straightforward way to run LLaVA on a local machine without needing an external API, enhancing privacy.
   - **Setup**: Install Ollama at [Ollama's official site](https://ollama.com). Once installed, run Ollama to serve LLaVA as follows:
     ```bash
     ollama run llava
     ```
   - **Endpoint**: Ollama will create an endpoint at `http://localhost:11434`, which the application uses to interact with the LLaVA model.

### Step 4: Run the Application
To start the application, execute:
```bash
python app.py
```

### Step 5: Usage
1. Set the interval, activities, and blacklisted words in the settings.
2. Click "Start Monitoring" to begin.
3. The app will periodically capture the screen, analyze activities, and alert you if you’re distracted.

### Additional Notes

- **Audio Alerts**: Ensure the `Radar.mp3` file (or your custom audio) is available in the directory specified in `config.json`.

## Troubleshooting

- **Screen Capture Issues**: Ensure you have the correct permissions and screen capture software.
- **Audio Alerts**: Ensure audio files are accessible and compatible with `playsound`.