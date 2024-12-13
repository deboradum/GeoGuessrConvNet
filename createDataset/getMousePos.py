import time
import pyautogui

def get_mouse_position():
    """
    Continuously print the current mouse position.
    Press Ctrl+C to stop the tracking.
    """
    try:
        print("Mouse Position Tracker (Press Ctrl+C to exit)")
        print("Move your mouse around to see its coordinates.")
        while True:
            # Get current mouse position
            x, y = pyautogui.position()

            # Print the coordinates
            print(f"Mouse Position: X = {x}, Y = {y}", end="\r")

            # Small delay to reduce CPU usage
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nTracking stopped.")

get_mouse_position()
