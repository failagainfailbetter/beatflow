import random
import re

INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "11": [0, 7, 10, 14, 17],
    "13": [0, 4, 7, 10, 14, 21]
}

NOTE_MAP = {'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'F':5, 
            'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11}

DURATIONS = {
    '1n': 4.0,
    '2n': 2.0,
    '4n': 1.0,
    '8n': 0.5,
    '16n': 0.25,
    '32n': 0.125,
    '4t': 4.0 / 3.0,
    '8t': 1.0 / 3.0,
    '16t': 0.5 / 3.0
}

def parse_complex_chord(chord_name, default_octave=4):
    chord_name = chord_name.strip()
    
    root_match = re.match(r"^([A-G][#b]?)", chord_name)
    if not root_match:
        return [60, 64, 67]
    
    root_str = root_match.group(1)
    root_val = NOTE_MAP.get(root_str, 0)
    
    current_octave = default_octave
    if root_val >= 5:
        current_octave -= 1
        
    root_midi = root_val + (current_octave + 1) * 12
    suffix = chord_name[len(root_str):]
    
    if suffix == "" or suffix == "5": quality = "maj"
    elif suffix == "m": quality = "min"
    elif suffix == "+": quality = "aug"
    elif "maj9" in suffix: quality = "maj9"
    elif "min9" in suffix or "m9" in suffix: quality = "min9"
    elif "maj7" in suffix: quality = "maj7"
    elif "min7" in suffix or "m7" in suffix: quality = "min7"
    elif "7" in suffix: quality = "7"
    elif "9" in suffix: quality = "9"
    elif "13" in suffix: quality = "13"
    elif "dim" in suffix: quality = "dim"
    elif "sus4" in suffix: quality = "sus4"
    else: quality = "maj"
    
    intervals = INTERVALS.get(quality, INTERVALS["maj"])
    notes = [root_midi + i for i in intervals]
    
    final_notes = []
    
    for i, note in enumerate(notes):
        if i == 0:
            final_notes.append(note - 12)
        elif note > root_midi + 12:
            final_notes.append(note) 
        else:
            final_notes.append(note)
            
    return sorted(final_notes)

def create_note_event(pitch, start, dur, velocity, track_id):
    return {"note": int(pitch), "start": start, "duration": dur, "velocity": int(velocity), "track_id": track_id}

def get_groove_offset(start_time, groove_type):
    offset = 0.0
    jitter = random.uniform(-0.005, 0.005)
    
    step_index = int(round(start_time / 0.25))
    
    if groove_type == "swing":
        if step_index % 2 != 0: offset += 0.04
    elif groove_type == "heavy_swing" or groove_type == "shuffle":
        if step_index % 2 != 0: offset += 0.08
    elif groove_type == "drunk":
        offset += random.uniform(-0.02, 0.02)
        if step_index % 2 != 0: offset += 0.03
    elif groove_type == "laid_back":
        offset += 0.02 
    elif groove_type == "rushed":
        offset -= 0.01

    return offset + jitter

def parse_duration_stream(stream):
    notes = []
    current_time_beats = 0.0
    
    if not isinstance(stream, list):
        return notes
        
    for item in stream:
        match = re.match(r'^([A-Za-z0-9_\-\.]+?)(1n|2n|4n|8n|16n|32n|4t|8t|16t)$', str(item).strip())
        
        if not match:
            current_time_beats += 0.25
            continue
            
        event_char = match.group(1)
        dur_str = match.group(2)
        duration_beats = DURATIONS[dur_str]
        
        if event_char not in ['.', 'rest']:
            velocity = 100
            if event_char == 'X': velocity = 127
            elif event_char == 'g': velocity = 50
            
            notes.append({
                "start_time": current_time_beats,
                "duration": duration_beats,
                "event": event_char,
                "velocity": velocity
            })
            
        current_time_beats += duration_beats
        
    return notes

def parse_drum_grid(stream, track_id, midi_note, groove_type="straight"):
    events = []
    parsed_notes = parse_duration_stream(stream)
    
    for n in parsed_notes:
        vel = 90
        if n["event"] == 'X': vel = 120
        elif n["event"] == 'x': vel = 100
        elif n["event"] == 'g': vel = 50
        
        timing_offset = get_groove_offset(n["start_time"], groove_type)
        if groove_type == "drunk" and midi_note == 38:
            timing_offset += 0.03
            
        start_time = n["start_time"] + timing_offset
        if start_time < 0: start_time = 0
            
        events.append(create_note_event(midi_note, start_time, n["duration"] * 0.95, vel, track_id))
        
    return events

def parse_harmonic_grid(stream, chord_name, instrument_type, track_id, groove_type="straight"):
    events = []
    chord_notes = parse_complex_chord(chord_name, default_octave=3)
    parsed_notes = parse_duration_stream(stream)
    
    for n in parsed_notes:
        if n["event"] == '-':
            continue
            
        target_idx = 0
        num_match = re.search(r'\d', n["event"])
        if num_match:
            val = num_match.group()
            if val == '1': target_idx = 0
            elif val == '3': target_idx = 1
            elif val == '5': target_idx = 2
            elif val == '7': target_idx = 3
            elif val == '9': target_idx = 4
            
        if target_idx < len(chord_notes):
            current_note = chord_notes[target_idx]
        else:
            current_note = chord_notes[0]
            
        final_pitch = current_note
        if instrument_type == "bass":
            final_pitch -= 12
            if final_pitch < 36: final_pitch += 12
            
        start_offset = get_groove_offset(n["start_time"], groove_type)
        final_start = n["start_time"] + start_offset
        if final_start < 0: final_start = 0
            
        current_vel = random.randint(85, 105)
        
        events.append(create_note_event(final_pitch, final_start, n["duration"] * 0.95, current_vel, track_id))
        
    return events

def parse_chord_comping(stream, chord_name, track_id, groove_type="straight"):
    events = []
    chord_notes = parse_complex_chord(chord_name, default_octave=4)
    parsed_notes = parse_duration_stream(stream)
    
    for n in parsed_notes:
        if n["event"].lower() == 'x':
            offset = get_groove_offset(n["start_time"], groove_type)
            final_start = n["start_time"] + offset
            if final_start < 0: final_start = 0
            vel = random.randint(80, 95)
            
            for note in chord_notes:
                events.append(create_note_event(note, final_start, n["duration"] * 0.95, vel, track_id))
                
    return events