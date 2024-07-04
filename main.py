import os
import re
import sys
import argparse
import json
import tempfile
import numpy as np
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from tqdm import tqdm

client = ElevenLabs(api_key=os.environ["ELEVEN_API_KEY"])


def parse_dialogue_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    lines = content.split("\n")
    dialogue = []
    characters = set()
    for line in lines:
        match = re.match(r"(\w+):\s*(.+)", line.strip())
        if match:
            character, text = match.groups()
            dialogue.append((character, text))
            characters.add(character)

    return dialogue, list(characters)


def get_available_voices():
    response = client.voices.get_all()
    return [voice.name for voice in response.voices]


def assign_voices_to_characters(characters, available_voices):
    voice_map = {}
    for i, character in enumerate(characters):
        voice_map[character] = available_voices[i % len(available_voices)]
    return voice_map


def generate_character_audio(character, text, voice_map):
    voice = voice_map[character]
    audio = client.generate(
        text=text, voice=voice, model="eleven_multilingual_v2", stream=True
    )
    return audio


def save_temp_audio(audio):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        for chunk in audio:
            temp_file.write(chunk)
    return temp_file.name


def create_conversation_audio(dialogue, voice_map, pause_mean, pause_stdev):
    conversation = AudioSegment.empty()

    for character, text in tqdm(dialogue, desc="Generating audio"):
        audio = generate_character_audio(character, text, voice_map)
        temp_file = save_temp_audio(audio)
        audio_segment = AudioSegment.from_mp3(temp_file)
        os.unlink(temp_file)  # Remove the temporary file

        pause_duration = int(np.random.normal(pause_mean, pause_stdev))
        pause_duration = max(0, pause_duration)  # Ensure non-negative duration

        conversation += audio_segment + AudioSegment.silent(duration=pause_duration)

    return conversation


def main(args):
    dialogue, characters = parse_dialogue_file(args.input_file)
    available_voices = get_available_voices()

    if args.speaker_map:
        with open(args.speaker_map, "r") as f:
            custom_voice_map = json.load(f)
        invalid_voices = set(custom_voice_map.values()) - set(available_voices)
        if invalid_voices:
            print(
                f"Warning: The following voices in the custom map are not available: {', '.join(invalid_voices)}"
            )
            print("Available voices:", ", ".join(available_voices))
            sys.exit(1)
        voice_map = custom_voice_map
    else:
        voice_map = assign_voices_to_characters(characters, available_voices)

    print("Voice assignments:")
    for character, voice in voice_map.items():
        print(f"{character}: {voice}")

    conversation = create_conversation_audio(
        dialogue, voice_map, args.pause_mean, args.pause_stdev
    )

    output_file = f"{args.input_file.rsplit('.', 1)[0]}_output.wav"
    conversation.export(output_file, format="wav")

    print(f"\nConversation audio generated and saved as '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate game dialogue audio using ElevenLabs API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to the dialogue text file")
    parser.add_argument(
        "-m",
        "--map",
        dest="speaker_map",
        help="Path to a JSON file containing a custom speaker-to-voice map",
    )
    parser.add_argument(
        "--pause-mean",
        type=int,
        default=500,
        help="Mean duration of pause (in ms) between each line of dialogue",
    )
    parser.add_argument(
        "--pause-stdev",
        type=int,
        default=100,
        help="Standard deviation of pause duration (in ms)",
    )
    args = parser.parse_args()

    main(args)
