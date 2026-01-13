import os
import sys
import ast

# --- Platform Specific Input Handling ---
try:
    import msvcrt
    PLATFORM = 'win'
except ImportError:
    import tty
    import termios
    PLATFORM = 'unix'

def get_key():
    """Reads a keypress and returns a unified code (up, down, enter, etc.)."""
    if PLATFORM == 'win':
        # Windows msvcrt
        key = msvcrt.getch()
        if key == b'\xe0':  # Arrow key prefix
            key = msvcrt.getch()
            if key == b'H': return 'up'
            if key == b'P': return 'down'
            if key == b'K': return 'left'
            if key == b'M': return 'right'
        elif key == b'\r': return 'enter'
        elif key == b'\x08': return 'backspace'
        elif key == b'\x1b': return 'esc'
        else:
            try: return key.decode()
            except: return ''
    else:
        # Linux/Unix termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                seq = sys.stdin.read(2)
                if seq == '[A': return 'up'
                if seq == '[B': return 'down'
                if seq == '[D': return 'left'
                if seq == '[C': return 'right'
                return 'esc'
            elif ch == '\r': return 'enter'
            elif ch == '\x7f': return 'backspace'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ''

CONFIG_FILE = 'config.cfg'

# Define Categories for the Menu
CATEGORIES = {
    "Main Settings": [
        "video_source", "model", "confidence_threshold", "search_for", 
        "no_display", "output_path", "video_save", "device", "verbose", 
        "time_out", "max_history", "show_arrow", "show_trail"
    ],
    "Socket": ["use_socket", "socket_port"],
    "GPIO": ["use_gpio", "gpio_pin", "setmode"],
    "Performance": ["skip_frames", "resize_width"]
}

def load_raw_config():
    """Reads the config file lines."""
    if not os.path.exists(CONFIG_FILE):
        return []
    with open(CONFIG_FILE, 'r') as f:
        return f.readlines()

def save_raw_config(lines):
    """Writes lines back to the config file."""
    with open(CONFIG_FILE, 'w') as f:
        f.writelines(lines)

def change_setting(key, new_value):
    """Updates a specific key in the config file while preserving comments."""
    lines = load_raw_config()
    new_lines = []
    found = False
    
    for line in lines:
        # Check if this line contains the key assignment
        clean = line.split('#')[0].strip()
        if '=' in clean:
            k, v = clean.split('=', 1)
            if k.strip() == key:
                # Preserve existing comment if present
                comment = ""
                if '#' in line:
                    comment = "  #" + line.split('#', 1)[1].strip()
                new_lines.append(f"{key} = {new_value}{comment}\n")
                found = True
                continue
        new_lines.append(line)
    
    if not found:
        # Append new key if not found
        new_lines.append(f"{key} = {new_value}\n")
    
    save_raw_config(new_lines)

def get_current_settings():
    """Returns a dict of current settings for display."""
    settings = {}
    lines = load_raw_config()
    for line in lines:
        clean = line.split('#')[0].strip()
        if '=' in clean:
            k, v = clean.split('=', 1)
            settings[k.strip()] = v.strip()
    return settings

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_editor():
    current_category = None
    selected_index = 0
    
    while True:
        clear_screen()
        settings = get_current_settings()
        
        if current_category is None:
            # --- Main Menu (Categories) ---
            print("========================================")
            print("      Configuration Editor              ")
            print("========================================")
            print("Select a category (Use Arrow Keys + Enter):")
            print("----------------------------------------")
            
            cats = list(CATEGORIES.keys()) + ["Exit"]
            for i, cat in enumerate(cats):
                prefix = " > " if i == selected_index else "   "
                print(f"{prefix}{cat}")
                
            key = get_key()
            if key == 'up':
                selected_index = max(0, selected_index - 1)
            elif key == 'down':
                selected_index = min(len(cats) - 1, selected_index + 1)
            elif key == 'enter':
                choice = cats[selected_index]
                if choice == 'Exit':
                    break
                current_category = choice
                selected_index = 0 # Reset index for sub-menu
        else:
            # --- Sub Menu (Settings in Category) ---
            keys_in_cat = CATEGORIES[current_category]
            
            print(f"=== {current_category} ===")
            print("Use Arrows to move, Enter to edit/toggle, Esc/Backspace to go back")
            print("-----------------------------------------------------------")
            
            for i, key_name in enumerate(keys_in_cat):
                prefix = " > " if i == selected_index else "   "
                val = settings.get(key_name, "Not Set")
                print(f"{prefix}{key_name:<25} = {val}")
            
            key = get_key()
            if key == 'up':
                selected_index = max(0, selected_index - 1)
            elif key == 'down':
                selected_index = min(len(keys_in_cat) - 1, selected_index + 1)
            elif key in ['esc', 'backspace']:
                current_category = None
                selected_index = 0
            elif key == 'enter':
                # --- Edit Logic ---
                key_to_edit = keys_in_cat[selected_index]
                raw_val = settings.get(key_to_edit, "")
                
                # Determine type
                try:
                    curr_val = ast.literal_eval(raw_val)
                except (ValueError, SyntaxError):
                    if raw_val.lower() == 'true': curr_val = True
                    elif raw_val.lower() == 'false': curr_val = False
                    else: curr_val = raw_val

                if isinstance(curr_val, bool):
                    # Toggle Boolean immediately
                    change_setting(key_to_edit, str(not curr_val))
                else:
                    # Input for other types
                    print(f"\nEditing '{key_to_edit}'")
                    new_input = input(f"Enter new value (Current: {raw_val}): ").strip()
                    if new_input:
                        # Auto-quote strings if user didn't
                        if isinstance(curr_val, str) and not (new_input.startswith(('"', "'")) and new_input.endswith(('"', "'"))):
                             # Check if it looks like a number, if so, don't quote unless original was string
                             try:
                                 float(new_input)
                                 # It's a number. If the original was strictly a string (like "0" for video source), 
                                 # we might want to keep it quoted, but config parser handles both usually.
                                 # For safety, if original was string, we quote.
                                 if isinstance(curr_val, str):
                                     new_input = f'"{new_input}"'
                             except ValueError:
                                 # Not a number, definitely quote it
                                 new_input = f'"{new_input}"'
                        
                        change_setting(key_to_edit, new_input)

if __name__ == "__main__":
    run_editor()