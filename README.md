# Real-Time Vision AI Assistant with Live Stream

This repository contains a Streamlit application that uses a Vision Model API to analyze live video streams from your webcam in real time. The app captures frames from your webcam, processes them using an external Vision Model via OpenRouter, and displays both the live feed and analysis results dynamically.

## Features

- **Real-Time Webcam Feed:**  
  Captures and displays live video from your default webcam.
  
- **Frame Analysis:**  
  Periodically analyzes selected frames (every 30th frame) to generate a description of the scene.
  
- **Dynamic Updates:**  
  Continuously updates the interface with the latest analysis results alongside the live stream.
  
- **Customizable Prompts:**  
  Allows users to specify a custom text prompt for the Vision Model.

- **Easy API Integration:**  
  Integrates with the OpenRouter Vision Model API for image analysis.

## Requirements

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [Requests](https://docs.python-requests.org/)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/vision-ai-assistant.git
   cd vision-ai-assistant
