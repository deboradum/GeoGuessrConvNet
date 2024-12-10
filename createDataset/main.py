import os
import time
import pyautogui
from PIL import ImageGrab

MAP_X, MAP_Y = 1779, -183
GUESS_X, GUESS_Y = 1789, -40
CENTER_X, CENTER_Y = 1853, -677


def take_screenshot(directory):
    i = 1
    while os.path.exists(os.path.join(directory, f"{i}.png")):
        i += 1

    screenshot = ImageGrab.grab(bbox=(0, -1000, 1600, -100))
    screenshot_path = os.path.join(directory, f"{i}.png")
    screenshot.save(screenshot_path)

    print(f"Screenshot saved to {screenshot_path}")


def simulate_mouse_click(x, y):
    try:
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()
    except pyautogui.FailSafeError:
        print("Mouse moved to a fail-safe corner. Click cancelled.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    curr_state = "Indiana"
    counter = 0
    while True:
        os.makedirs(curr_state, exist_ok=True)
        take_screenshot(curr_state)

        simulate_mouse_click(MAP_X, MAP_Y)  # open map
        simulate_mouse_click(MAP_X, MAP_Y)  # Click map
        pyautogui.press("space")  # Make guess
        pyautogui.moveTo(CENTER_X, CENTER_Y, duration=0.1)  # Prevent map opening again
        pyautogui.press("space")  # Next round

        time.sleep(0.05)
        pyautogui.press("space")

        counter += 1
        if counter % 5 == 0:
            time.sleep(0.05)
            pyautogui.press("space")
        time.sleep(0.7)  # Wait for next round
