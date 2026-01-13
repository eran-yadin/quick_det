import os
import sys
import subprocess

# Import your modules
import configure_zone
import test
import config_editor

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

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    options = [
        "Start Detection App (main.py)",
        "Configure Zone (Camera Setup)",
        "Test Camera & Viewport",
        "Edit Configuration",
        "Quit"
    ]
    selected_index = 0

    while True:
        clear_screen()
        print("========================================")
        print("      Human Detection Pro - Launcher    ")
        print("========================================")
        print("Use Arrow Keys to move, Enter to select")
        print("----------------------------------------")
        
        for i, option in enumerate(options):
            prefix = " > " if i == selected_index else "   "
            print(f"{prefix}{option}")
            
        print("========================================")
        
        key = get_key()
        
        if key == 'up':
            selected_index = max(0, selected_index - 1)
        elif key == 'down':
            selected_index = min(len(options) - 1, selected_index + 1)
        elif key == 'enter':
            choice = options[selected_index]
            
            if choice == "Start Detection App (main.py)":
                print("\n[INFO] Starting Main App...")
                # Run main.py as a subprocess to ensure clean environment 
                # and avoid multiprocessing conflicts
                try:
                    subprocess.run([sys.executable, "main.py"])
                except KeyboardInterrupt:
                    pass
                input("\nApp closed. Press Enter to return...")
                
            elif choice == "Configure Zone (Camera Setup)":
                print("\n[INFO] Starting Zone Configuration...")
                try:
                    configure_zone.configure_detector()
                except Exception as e:
                    print(f"Error: {e}")
                input("\nConfiguration finished. Press Enter to return...")
                
            elif choice == "Test Camera & Viewport":
                print("\n--- Camera Diagnostics ---")
                try:
                    print("1. Checking Camera Access...")
                    test.test_camera_access()
                    print(">> Camera Access: PASS")
                    
                    ans = input("2. Test Live Viewport? (y/n): ")
                    if ans.lower() == 'y':
                        test.test_view_port()
                except Exception as e:
                    print(f"\n[FAIL] Test failed: {e}")
                
                input("\nTests complete. Press Enter to return...")
                
            elif choice == "Edit Configuration":
                config_editor.run_editor()
                
            elif choice == "Quit":
                print("Goodbye!")
                break

if __name__ == "__main__":
    main_menu()