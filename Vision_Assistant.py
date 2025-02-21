import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import requests
import json
from PIL import Image
import base64
from io import BytesIO

# OpenRouter API endpoint for the Vision Model
LVM_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def openrouter_request(api_url: str, headers: dict, data: dict) -> dict:
    """
    Sends a POST request to the specified API URL with given headers and data.
    """
    try:
        response = requests.post(url=api_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}\n{response.text}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    return None

def analyze_image(lvm_api_key: str, image_url: str, prompt: str = "What is in this image?") -> str:
    """
    Analyzes an image using the Vision Model API.
    """
    headers = {
        "Authorization": f"Bearer {lvm_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "google/gemini-2.0-pro-exp-02-05:free",  # Consistent model across API calls
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
    }
    result = openrouter_request(LVM_API_URL, headers, data)
    if result and "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    return "No description available."

def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image to a base64 encoded JPEG data URL.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

class VideoAnalyzer(VideoTransformerBase):
    def __init__(self, lvm_api_key: str, prompt: str):
        self.lvm_api_key = lvm_api_key
        self.prompt = prompt
        self.frame_index = 0
        self.all_results = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to numpy array
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        pil_img = Image.fromarray(rgb_img)  # Convert to PIL Image

        # Analyze every 30th frame
        if self.frame_index % 30 == 0:
            img_url = image_to_base64(pil_img)
            with st.spinner(f"Analyzing frame {self.frame_index}..."):
                description = analyze_image(self.lvm_api_key, img_url, prompt=self.prompt.strip())
                self.all_results.append(f"Frame {self.frame_index}: {description}")
                st.markdown(
                    "<br>".join(self.all_results),  # Display all results separated by line breaks
                    unsafe_allow_html=True
                )
        self.frame_index += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Real-Time Vision AI Assistant with Live Stream")
    st.markdown(
        """
        This application uses the **Vision Model** to analyze live video streams from your webcam in real-time.
        The live webcam feed is displayed within the app, and all analysis results are shown dynamically.
        """
    )

    # Sidebar for API Key input
    st.sidebar.header("API Key")
    lvm_api_key = st.sidebar.text_input("Enter your Vision Model API Key:", type="password")
    if not lvm_api_key:
        st.sidebar.warning("Please enter your API key to proceed.")
        st.stop()

    # Validate API Key with a test request
    test_data = {
        "model": "google/gemini-2.0-pro-exp-02-05:free",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Test"}]}],
    }
    headers = {"Authorization": f"Bearer {lvm_api_key}", "Content-Type": "application/json"}
    if openrouter_request(LVM_API_URL, headers, test_data) is None:
        st.error("Invalid API key or connection issue. Please check your credentials and network.")
        st.stop()

    # Real-Time Video Analysis Section
    st.header("Real-Time Video Analysis with Live Stream")
    custom_prompt_realtime = st.text_input(
        "Enter a prompt for real-time analysis (e.g., 'What is happening right now?'):",
        value="What is happening in this frame?"
    )
    if st.button("Start Real-Time Analysis"):
        if lvm_api_key:
            st.info("Starting real-time analysis using your webcam...")
            ctx = webrtc_streamer(
                key="example",
                video_transformer_factory=lambda: VideoAnalyzer(lvm_api_key, custom_prompt_realtime),
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )
        else:
            st.error("Please enter your API key to proceed.")

if __name__ == "__main__":
    main()
