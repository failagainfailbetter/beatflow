import os
import json
import random
import re
from openai import OpenAI
from dotenv import load_dotenv
from services.music_engine import (
    parse_drum_grid, parse_harmonic_grid, parse_chord_comping
)

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model_name = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b:free")

_missing_llm_vars = [name for name, val in [
    ("OPENROUTER_API_KEY", api_key),
    ("LLM_BASE_URL", base_url),
] if not val]

if _missing_llm_vars:
    print(f"WARNING: Missing required environment variables: {', '.join(_missing_llm_vars)}. "
          "The 'generate beats' feature will not work until these are set.")
    client = None
else:
    client = OpenAI(base_url=base_url, api_key=api_key)
MODEL_NAME = model_name

# Probability that a non-last bar becomes a fill bar (adds musical interest)
FILL_BAR_PROBABILITY = 0.25
# Probability that a rest event becomes a ghost note during humanization
REST_TO_GHOST_MULTIPLIER = 0.4
# Probability that a hit becomes an accent (vs. a ghost) during humanization
ACCENT_PROBABILITY = 0.5

COMBINED_PROMPT = """
You are a Senior Music Director AND World-Class Rhythm Composer.
Your Goal: Create a complete, full-length song composition with structure AND MIDI patterns in ONE response.

User Request: "{vibe}"

SONG STRUCTURE REQUIREMENTS:
- Generate 8-12 sections forming a complete arrangement: intro, verse, pre-chorus, chorus, bridge, build, drop, breakdown, outro
- Each section should be 4-8 bars long (aim for 8 bars on main sections)
- Plan a dynamic energy arc: sparse intro → building verse → peak chorus → atmospheric breakdown → drop → outro
- Use rich, sophisticated harmony: extended chords like Cmaj7, F#m9b5, Am11, Dm9, G13, Bb13sus4, Fmaj9, Em7b9

FOR EACH SECTION, provide complete MIDI patterns using Duration Stream Notation:
- Each array represents exactly ONE bar (4.0 beats total)
- Format per event: <Event><Duration>
- Events: 'x' (Hit), 'X' (Accent/strong hit), 'g' (Ghost note, quiet), '.' (Rest/silence), scale degrees for bass only: '1','3','5','7','9', '-' (Sustain/tie)
- Durations: '1n'=4 beats, '2n'=2 beats, '4n'=1 beat, '8n'=0.5, '16n'=0.25, '8t'=0.333 (triplet eighth), '16t'=0.167
- CRITICAL: durations in each array MUST sum to exactly 4.0 beats

MUSICAL REQUIREMENTS:
1. Drums: Design genre-specific grooves. Snare typically on beats 2 and 4. Include ghost notes (g) for humanization and texture. Kick patterns should anchor the groove.
2. Bass: Use scale degrees (1=root, 3=third, 5=fifth, 7=seventh). Syncopate! Avoid quarter-note monotony. Walk between chord tones.
3. Keys/Piano: Use 'x'/'X' for chord hits only (no numbers). Syncopate the comping rhythm. Stabs, anticipations, offbeats - NO boring whole notes.
4. Fill patterns (kick_fill, snare_fill, hihat_fill, bass_fill, keys_fill): Make these busier/more complex than main patterns - fills add excitement at phrase endings.
5. Vary the groove field: "straight", "swing", "heavy_swing", "drunk", or "laid_back" - vary between sections for musical interest.
6. Energy contrast: sparse breakdown sections should have minimal drums and spacious piano; drop sections should be dense and powerful.

Return JSON ONLY with this structure (generate all 8-12 sections - DO NOT truncate):
{{
  "bpm": 120,
  "key": "A Minor",
  "sections": [
    {{
      "name": "Intro",
      "length": 8,
      "energy": "Low",
      "texture": "Sparse",
      "chords": ["Am11", "Fmaj9", "Cmaj7", "G13"],
      "groove": "straight",
      "kick_main":  ["<one bar of kick hits summing to 4 beats>"],
      "kick_fill":  ["<busier fill bar summing to 4 beats>"],
      "snare_main": ["<one bar of snare hits summing to 4 beats>"],
      "snare_fill": ["<busier fill bar summing to 4 beats>"],
      "hihat_main": ["<one bar of hihat hits summing to 4 beats>"],
      "hihat_fill": ["<busier fill bar summing to 4 beats>"],
      "bass_main":  ["<one bar using scale degrees summing to 4 beats>"],
      "bass_fill":  ["<walking/busier bass fill summing to 4 beats>"],
      "keys_main":  ["<one bar of chord hits (x/X) summing to 4 beats>"],
      "keys_fill":  ["<syncopated fill comping summing to 4 beats>"]
    }},
    {{ "name": "Verse 1", "length": 8, ... }},
    {{ "name": "Pre-Chorus", "length": 4, ... }},
    {{ "name": "Chorus", "length": 8, ... }},
    {{ "name": "Verse 2", "length": 8, ... }},
    {{ "name": "Bridge", "length": 8, ... }},
    {{ "name": "Build", "length": 4, ... }},
    {{ "name": "Drop", "length": 8, ... }},
    {{ "name": "Breakdown", "length": 4, ... }},
    {{ "name": "Outro", "length": 4, ... }}
  ]
}}
"""

def get_json(prompt, model=MODEL_NAME):
    if _missing_llm_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(_missing_llm_vars)}. "
            "Please set OPENROUTER_API_KEY and LLM_BASE_URL."
        )
    messages = [
        {"role": "system", "content": "You are a JSON-only response bot."}, 
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.9, response_format={"type": "json_object"})
        content = resp.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"LLM Error: {e}")
        raise

def apply_random_spice(stream, probability=0.15):
    if not isinstance(stream, list):
        return stream

    new_stream = []
    for item in stream:
        match = re.match(r'^([A-Za-z0-9_\-\.]+?)(1n|2n|4n|8n|16n|32n|4t|8t|16t)$', str(item).strip())
        if not match:
            new_stream.append(item)
            continue

        event_char = match.group(1)
        dur_str = match.group(2)

        new_event = event_char
        if event_char == '.' and random.random() < (probability * REST_TO_GHOST_MULTIPLIER):
            new_event = 'g'
        elif event_char == 'x' and random.random() < probability:
            new_event = 'X' if random.random() > ACCENT_PROBABILITY else 'g'

        new_stream.append(f"{new_event}{dur_str}")

    return new_stream


def generate_section_clips(section_data, bpm, track_ids):
    chords = section_data.get("chords", ["Am7"])
    length = section_data.get("length", 4)
    groove_type = section_data.get("groove", "straight").lower()

    print(f"  > Groove: {groove_type} | BPM: {bpm} | Bars: {length}")

    clips = {"kick": [], "snare": [], "hat": [], "bass": [], "piano": []}

    for bar in range(length):
        offset = bar * 4.0
        # Use fill patterns for the last bar of each section plus ~25% of other bars
        is_fill_bar = (bar == length - 1) or (bar != 0 and random.random() < FILL_BAR_PROBABILITY)
        suffix = "_fill" if is_fill_bar else "_main"

        def get_grid(instr):
            grid = section_data.get(f"{instr}{suffix}")
            if not grid:
                grid = section_data.get(f"{instr}_main")
            if not grid:
                grid = section_data.get(instr)
            return grid

        k_grid = get_grid("kick")
        if k_grid:
            if not is_fill_bar:
                k_grid = apply_random_spice(k_grid, 0.08)
            k = parse_drum_grid(k_grid, track_ids["kick"], 36, groove_type)
            for e in k:
                e["start"] += offset
                clips["kick"].append(e)

        s_grid = get_grid("snare")
        if s_grid:
            if not is_fill_bar:
                s_grid = apply_random_spice(s_grid, 0.08)
            s = parse_drum_grid(s_grid, track_ids["snare"], 38, groove_type)
            for e in s:
                e["start"] += offset
                clips["snare"].append(e)

        h_grid = get_grid("hihat")
        if not h_grid:
            h_grid = ["x8n", "x8n", "x8n", "x8n", "x8n", "x8n", "x8n", "x8n"]
        if not is_fill_bar:
            h_grid = apply_random_spice(h_grid, 0.15)
        h = parse_drum_grid(h_grid, track_ids["hat"], 42, groove_type)
        for e in h:
            e["start"] += offset
            clips["hat"].append(e)

        current_chord = chords[bar % len(chords)] if chords else "Am7"

        b_grid = get_grid("bass")
        if b_grid:
            b = parse_harmonic_grid(b_grid, current_chord, "bass", track_ids["bass"], groove_type)
            for e in b:
                e["start"] += offset
                clips["bass"].append(e)

        p_grid = get_grid("keys")
        if p_grid:
            p = parse_chord_comping(p_grid, current_chord, track_ids["piano"], groove_type)
            for e in p:
                e["start"] += offset
                clips["piano"].append(e)

    return clips


def generate_music_json(user_prompt: str):
    print(f"request: {user_prompt}")

    result = get_json(COMBINED_PROMPT.format(vibe=user_prompt))
    bpm = result.get("bpm", 90)
    sections = result.get("sections", [])

    if not sections:
        sections = [{"name": "Jam", "length": 4, "energy": "Medium", "chords": ["Cm7", "F9"], "groove": "straight"}]

    final_json = {
        "bpm": bpm,
        "tracks": [
            {"id": "t_piano", "instrument": "Electric Piano", "type": "instrument"},
            {"id": "t_bass", "instrument": "Finger Bass", "type": "instrument"},
            {"id": "t_kick", "instrument": "Kick", "type": "percussion"},
            {"id": "t_snare", "instrument": "Snare", "type": "percussion"},
            {"id": "t_hat", "instrument": "HiHat", "type": "percussion"},
        ],
        "clips": {},
        "arrangement": []
    }

    track_ids = {
        "piano": "t_piano", "bass": "t_bass",
        "kick": "t_kick", "snare": "t_snare", "hat": "t_hat"
    }

    curr_bar = 0

    for i, sec in enumerate(sections):
        print(f"Composing Section {i+1}: {sec.get('name')}...")
        section_clips = generate_section_clips(sec, bpm, track_ids)

        for instr_name, events in section_clips.items():
            if not events:
                continue

            unique_id = f"s{i}_{sec.get('name')}_{instr_name}".replace(" ", "_")

            final_json["clips"][unique_id] = events
            final_json["arrangement"].append({
                "section": sec.get("name"),
                "start_bar": curr_bar,
                "track_id": track_ids[instr_name],
                "clip_id": unique_id
            })

        curr_bar += sec.get("length", 2)

    return final_json