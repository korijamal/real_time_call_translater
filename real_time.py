import queue
import re
import pyaudio
import os
import wave
import io
from google.cloud import speech, translate_v2 as translate, texttospeech


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pipeline/credentials.json"


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

OPERATOR_LANGUAGE = "en"
caller_language = "ru"


p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

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

    responses = start_streaming(caller_language)  # default language for bootstrap

    for response in responses:
        for result in response.results:
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                print("🎙️ Heard:", transcript)

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
