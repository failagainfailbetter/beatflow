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
model_name = os.getenv("LLM_MODEL")

_missing_llm_vars = [name for name, val in [
    ("OPENROUTER_API_KEY", api_key),
    ("LLM_BASE_URL", base_url),
    ("LLM_MODEL", model_name),
] if not val]

if _missing_llm_vars:
    print(f"WARNING: Missing required environment variables: {', '.join(_missing_llm_vars)}. "
          "The 'generate beats' feature will not work until these are set.")
    client = None
else:
    client = OpenAI(base_url=base_url, api_key=api_key)
MODEL_NAME = model_name

STRUCTURE_PROMPT = """
You are a Senior Music Director.
Your Goal: Design a sophisticated song structure based on the user's request.

User Request: "{vibe}"
Key: {key}

Instructions:
1. **Analyze the Genre**: Determine the typical structure.
2. **Chain of Loops**: Create a progression of short sections (2-4 bars).
3. **Harmony**: Choose chords that fit the genre's color.

Return JSON ONLY:
{{
  "bpm": 120,
  "key": "A Minor",
  "sections": [
    {{"name": "Intro", "length": 4, "energy": "Low", "texture": "Sparse", "chords": ["Am7"]}},
    {{"name": "Groove A", "length": 4, "energy": "Medium", "texture": "Steady", "chords": ["Am7", "D9"]}},
    {{"name": "Breakdown", "length": 2, "energy": "Low", "texture": "Atmospheric", "chords": ["Fmaj7"]}},
    {{"name": "Drop/Chorus", "length": 4, "energy": "High", "texture": "Busy", "chords": ["Am7", "G", "F", "Em7"]}}
  ]
}}
"""

PATTERN_PROMPT = """
You are a World-Class Rhythm Composer.
Task: Compose MIDI duration streams for section "{section}".

Context:
- User Request / Genre: {vibe}
- BPM: {bpm}
- Energy: {energy}
- Chords: {chords}

**Duration Stream Notation**:
You MUST output arrays of strings representing musical events and their exact durations.
Format: <Event><Duration>
- Events: 'x' (Hit/Play), 'X' (Accent), 'g' (Ghost), '.' (Rest), '1'/'3'/'5'/'7' (Scale degrees - FOR BASS ONLY), '-' (Sustain).
- Durations: '1n', '2n', '4n', '8n', '16n', '8t' (Eighth Triplet, 3 fit in 1 beat), '16t'.
- Sum of durations in each array MUST EXACTLY equal 4.0 beats (1 Bar).

**CRITICAL INSTRUCTIONS FOR INSTRUMENTS**:
1. **Piano/Keys**: You MUST use 'x' or 'X' to trigger chords (e.g., "x8n", "x8t"). DO NOT use numbers for piano. Think deeply about the comping rhythm! Do NOT just lay boring whole notes. You MUST independently design the rhythm (syncopation, triplets, stabs, laid-back chords) based on the user's vibe!
2. **Bass**: Use '1', '3', '5', etc. for scale degrees (e.g., "1_8n", ".16n", "5_16n").
3. **Drums**: Design the groove based on the genre (e.g., four-on-the-floor, boom-bap, trap rolls).

**JSON OUTPUT FORMAT (NO EXAMPLES PROVIDED)**:
I am NOT giving you any rhythm examples because you must THINK FOR YOURSELF. 
Replace all "<generate_array_here>" placeholders with your own original duration stream arrays.

Return JSON ONLY:
{{
  "analysis": "Explain your rhythm design here. E.g., 'User wants triplet piano, so I MUST use 8t or 16t for keys_main. I will make the bass syncopated...'",
  "groove": "straight or swing",
  "kick_main":  ["<generate_array_here>"], 
  "kick_fill":  ["<generate_array_here>"],
  "snare_main": ["<generate_array_here>"],
  "snare_fill": ["<generate_array_here>"],
  "hihat_main": ["<generate_array_here>"],
  "hihat_fill": ["<generate_array_here>"],
  "bass_main":  ["<generate_array_here>"],
  "bass_fill":  ["<generate_array_here>"],
  "keys_main":  ["<generate_array_here>"],
  "keys_fill":  ["<generate_array_here>"]
}}
"""

def get_json(prompt, model=MODEL_NAME):
    if _missing_llm_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(_missing_llm_vars)}. "
            "Please set OPENROUTER_API_KEY, LLM_BASE_URL, and LLM_MODEL."
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

def apply_random_spice(stream, probability=0.1):
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
        if event_char == '.' and random.random() < (probability * 0.3):
            new_event = 'g'
        elif event_char == 'x' and random.random() < probability:
            new_event = 'X' if random.random() > 0.5 else 'x'
            
        new_stream.append(f"{new_event}{dur_str}")
        
    return new_stream

def generate_section_clips(section_data, vibe, bpm, track_ids):
    sec_name = section_data.get("name", "Section")
    chords = section_data.get("chords", [])
    length = section_data.get("length", 2)
    energy = section_data.get("energy", "Medium")
    texture = section_data.get("texture", "Steady")
    
    prompt = PATTERN_PROMPT.format(
        section=sec_name, 
        vibe=vibe, 
        bpm=bpm, 
        energy=energy, 
        texture=texture, 
        chords=str(chords)
    )
    patterns = get_json(prompt)
    
    if not patterns: patterns = {}
    
    analysis = patterns.get("analysis", "No analysis provided.")
    groove_type = patterns.get("groove", "straight").lower()
    
    print(f"  > Thought: {analysis}")
    print(f"  > Groove: {groove_type} | BPM: {bpm}")

    clips = {"kick": [], "snare": [], "hat": [], "bass": [], "piano": []}
    
    for bar in range(length):
        offset = bar * 4.0
        is_fill_bar = (bar == length - 1)
        suffix = "_fill" if is_fill_bar else "_main"
        
        def get_grid(instr):
            grid = patterns.get(f"{instr}{suffix}")
            if not grid: grid = patterns.get(f"{instr}_main")
            if not grid: grid = patterns.get(instr) 
            return grid

        k_grid = get_grid("kick")
        if k_grid:
            if not is_fill_bar: k_grid = apply_random_spice(k_grid, 0.05)
            k = parse_drum_grid(k_grid, track_ids["kick"], 36, groove_type)
            for e in k: e["start"] += offset; clips["kick"].append(e)

        s_grid = get_grid("snare")
        if s_grid:
            if not is_fill_bar: s_grid = apply_random_spice(s_grid, 0.05)
            s = parse_drum_grid(s_grid, track_ids["snare"], 38, groove_type)
            for e in s: e["start"] += offset; clips["snare"].append(e)
            
        h_grid = get_grid("hihat")
        if not h_grid: h_grid = ["x8n", "x8n", "x8n", "x8n", "x8n", "x8n", "x8n", "x8n"]
        if not is_fill_bar: h_grid = apply_random_spice(h_grid, 0.1)
        h = parse_drum_grid(h_grid, track_ids["hat"], 42, groove_type)
        for e in h: e["start"] += offset; clips["hat"].append(e)
            
        current_chord = chords[bar % len(chords)]
        
        b_grid = get_grid("bass")
        if b_grid:
            b = parse_harmonic_grid(b_grid, current_chord, "bass", track_ids["bass"], groove_type)
            for e in b: e["start"] += offset; clips["bass"].append(e)
            
        p_grid = get_grid("keys")
        if p_grid:
            p = parse_chord_comping(p_grid, current_chord, track_ids["piano"], groove_type)
            for e in p: e["start"] += offset; clips["piano"].append(e)
            
    return clips

def generate_music_json(user_prompt: str):
    print(f"request: {user_prompt}")
    
    bp_data = get_json(STRUCTURE_PROMPT.format(vibe=user_prompt, key="Random"))
    bpm = bp_data.get("bpm", 90)
    sections = bp_data.get("sections", [])
    
    if not sections:
        sections = [{"name": "Jam", "length": 4, "energy": "Medium", "chords": ["Cm7", "F9"]}]
        
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
        section_clips = generate_section_clips(sec, user_prompt, bpm, track_ids)
        
        for instr_name, events in section_clips.items():
            if not events: continue
            
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