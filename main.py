import os
import streamlit as st
import whisper
import tempfile
import subprocess
import openai
import json
from mutagen import File


openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

model = whisper.load_model("base")


def generate_gpt_response(user_text, model_name="gpt-3.5-turbo-0125", print_output=False):
    print(f"received: {user_text}")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": user_text,
        }],
        model=model_name,
    )
    return response.choices[0].message.content


def get_create_date(audio_path):
    try:
        audio_file = File(audio_path)
        if audio_file and 'create_date' in audio_file.tags:
            return audio_file.tags.get('create_date')
        else:
            return None
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None


def process_file(audio_file):
    audio_filename = audio_file.name
    st.write(f"Processing file: {audio_filename}")

    if not os.path.exists('processed/'):
        os.makedirs('processed/')
        
    if f"{audio_filename}.json" in os.listdir('processed/'):
        st.success(f"{audio_filename} already processed")
        contents = json.loads(open(f"processed/{audio_filename}.json").read())
        st.write(contents)
        return 

    with tempfile.NamedTemporaryFile(suffix=".mp3" if audio_file.name.endswith(".m4a") else ".wav", delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        input_audio_path = temp_audio.name

    if input_audio_path.endswith(".m4a"):
        output_audio_path = input_audio_path.replace(".m4a", ".mp3")

        ffmpeg_cmd = f"ffmpeg -i {input_audio_path} -vn -ar 44100 -ac 2 -b:a 192k {output_audio_path}"
        subprocess.run(ffmpeg_cmd, shell=True)
        audio_path = output_audio_path  
    else:
        audio_path = input_audio_path

    create_date = get_create_date(audio_path)

    result = model.transcribe(audio_path, language="pt")
    st.markdown('### Transcription: ')
    st.write(result['text'])

    prompt = f"""
    Extract the following information from this text:
    Text: "{result['text']}"

    Beware that in this transcription there might be speech understanding errors like "Esse UV" actually meant to be "SUV" or "fume"(window tint) being interpreted as "fumei", etc...
    Do not add information that weren't said in the transcription, if any field has no mention in the transcription, leave it blank
    Format the result in this JSON format, it MUST be a valid JSON: 
        "car_type": SUV | Pickup | Sedan | Hatch | Truck | Bus | Motorcycle | Undefined | Other
        "car_model": "str"
        "car_color": str
        "infraction_description": str
        "license_plate": str
        "location": str
        "driver_info": male | female | undefined
        "infraction_severity": low | med | high
    """

    response = generate_gpt_response(prompt)

    try:
        processed_json = json.loads(response.strip())
        processed_json['transcription'] = result['text']
        processed_json['filename'] = audio_filename
        try:
            year, month, day, hour, minute, _ = audio_filename.split('_')
            processed_json['recording_date'] = f"{day}-{month}-{year} {hour}:{minute}"
        except:
            processed_json['recording_date'] = ""
        st.write("Processed Result (JSON):")
        st.write(processed_json)
        
        with open(f'processed/{audio_filename}.json', 'w') as f:
            json.dump(processed_json, f)
        st.success(f"{audio_filename} processed and saved")

    except json.JSONDecodeError:
        st.write("Error in extracting information. Please check the input text.")


def main():
    st.title("Upload Audio Files for Transcription")

    audio_files = st.file_uploader("Upload multiple audio files", type=["wav", "mp3", "m4a"], accept_multiple_files=True)

    if audio_files:
        for audio_file in audio_files:
            process_file(audio_file)


main()
