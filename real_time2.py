import queue
import re
import pyaudio
import os
import wave
import io
from google.cloud import speech, translate_v2 as translate, texttospeech
import subprocess
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "real_pipeline/credentials.json"

#

import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(
        f"Device {i}: {info['name']} (Input Channels: {info['maxInputChannels']}, Output Channels: {info['maxOutputChannels']})"
    )


def get_source_output_index(app_name="python"):
    output = subprocess.check_output(
        ["pactl", "list", "short", "source-outputs"]
    ).decode()
    for line in output.splitlines():
        if app_name in line:
            return line.split()[0]  # return source output index
    return None


def move_to_virtual_monitor(source_output_index):
    subprocess.run(
        ["pactl", "move-source-output", source_output_index, "VirtualSink.monitor"]
    )


for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if "pulse" in info["name"].lower():
        pulse_index = i
        print(f"Selected Pulse device index: {pulse_index} - {info['name']}")
        break

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

OPERATOR_LANGUAGE = "en"
caller_language = "en"


p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    input_device_index=pulse_index,  # 'pulse'
    frames_per_buffer=1024,
)

time.sleep(3)

index = get_source_output_index()
if index:
    print(f"🔁 Moving source-output {index} to VirtualSink.monitor")
    move_to_virtual_monitor(index)
else:
    print("❌ Could not find source-output for 'python'")

q = queue.Queue()


def mic_callback(in_data, frame_count, time_info, status):
    q.put(in_data)
    return (None, pyaudio.paContinue)


p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=mic_callback,
)
stream.start_stream()

# --- GOOGLE CLIENTS ---
speech_client = speech.SpeechClient()
translate_client = translate.Client()
tts_client = texttospeech.TextToSpeechClient()


# --- HELPERS ---
def is_sentence_complete(text):
    return bool(re.search(r"[.!?,؛।。？！،]$", text.strip()))


def play_audio(audio_content):
    wf = wave.open(io.BytesIO(audio_content), "rb")
    player = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )
    data = wf.readframes(1024)
    while data:
        player.write(data)
        data = wf.readframes(1024)
    player.stop_stream()
    player.close()


# TEXT TO SPEECH
def synthesize_and_play(text, lang_code):
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    input_text = texttospeech.SynthesisInput(text=text)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    response = tts_client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    play_audio(response.audio_content)


# --- STREAMING STT ---
def request_gen():
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)


# SPEECH TO TEXT
def start_streaming(language_code):

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_automatic_punctuation=True,
        speech_contexts=[
            speech.SpeechContext(
                phrases=["cardiac arrest", "gunshot", "asthma attack", "help me"],
                boost=20.0,  # Optional: Give higher weight to important phrases
            )
        ],
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    return speech_client.streaming_recognize(streaming_config, request_gen())


# --- MAIN LOOP ---
def main():
    print("🔊 Listening...")
    detected_lang = None
    transcript_buffer = ""

    responses = start_streaming(caller_language)

    for response in responses:
        for result in response.results:
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                print("🎙️ Heard:", transcript)
                #

                transcript_buffer += " " + transcript

                if not detected_lang:
                    detect_result = translate_client.translate(transcript_buffer)
                    detected_lang = detect_result["detectedSourceLanguage"]
                    print("🌍 Detected language:", detected_lang)

                if is_sentence_complete(transcript):
                    # Translate to operator language
                    translated = translate_client.translate(
                        transcript_buffer, target_language=OPERATOR_LANGUAGE
                    )
                    translated_text = translated["translatedText"]
                    print("🗣️ Translated:", translated_text)
                    # ! Send to front end

                    synthesize_and_play(translated_text, OPERATOR_LANGUAGE)
                    transcript_buffer = ""  # reset buffer


# --- RUN ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("🛑 Exiting...")
        stream.stop_stream()
        stream.close()
        p.terminate()
