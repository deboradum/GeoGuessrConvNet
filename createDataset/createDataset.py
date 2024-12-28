import os
import time
import torch
import hashlib
import pyautogui
import torchvision
from PIL import ImageGrab, Image

# Home1 screen
MAP_X, MAP_Y = 1779, -183
GUESS_X, GUESS_Y = 1789, -40
CENTER_X, CENTER_Y = 801, -634


def file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_image_hashes(directory, extensions=None):
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    hashes = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                try:
                    hashes.append(file_hash(filepath))
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")

    return hashes

model = torchvision.models.resnet18()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(),
    torch.nn.Linear(model.fc.in_features, 1),
)
best_state_dict = torch.load(
    "model.pth", map_location=torch.device("mps")
)
model.load_state_dict(best_state_dict)
model.eval()
def is_desired_image(screenshot_path):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(
                (896, 896),
            ),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    image = Image.open(screenshot_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()

    return probability >= 0.5


# Checks for two things:
# 1. If screenshot already exists
# 2. If screenshot succeeded (for example, exclude loading screens and the likes)
def is_legal_screenshot(screenshot_path, hashes):
    print("Checking for valid images")
    # Check if screenshot already exists
    img_hash = file_hash(screenshot_path)
    if img_hash in hashes:
        print("Screenshot already exists")
        return False
    hashes.append(img_hash)

    if not is_desired_image(screenshot_path):
        print("Not a desired image")
        return False

    return True


def take_screenshot(directory):
    i = 1
    while os.path.exists(os.path.join(directory, f"{i}.png")):
        i += 1

    screenshot = ImageGrab.grab(bbox=(0, -1000, 1600, -100))

    screenshot_path = os.path.join(directory, f"{i}.png")
    screenshot.save(screenshot_path)

    print(f"Screenshot saved to {screenshot_path}")

    return i, screenshot_path


def simulate_mouse_click(x, y):
    try:
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()
    except pyautogui.FailSafeError:
        print("Mouse moved to a fail-safe corner. Click cancelled.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    countries = [
        ("../countryDataset/train/Albania", "https://www.geoguessr.com/maps/albania"),
        ("../countryDataset/train/Argentina", "https://www.geoguessr.com/maps/argentina"),
        ("../countryDataset/train/Australia", "https://www.geoguessr.com/maps/australia"),
        ("../countryDataset/train/Austria", "https://www.geoguessr.com/maps/austria"),
        ("../countryDataset/train/Bangladesh", "https://www.geoguessr.com/maps/bangladesh"),
        ("../countryDataset/train/Belgium", "https://www.geoguessr.com/maps/belgium"),
        ("../countryDataset/train/Bhutan", "https://www.geoguessr.com/maps/bhutan"),
        ("../countryDataset/train/Bolivia", "https://www.geoguessr.com/maps/bolivia"),
        ("../countryDataset/train/Botswana", "https://www.geoguessr.com/maps/botswana"),
        ("../countryDataset/train/Brazil", "https://www.geoguessr.com/maps/brazil"),
        ("../countryDataset/train/Bulgaria", "https://www.geoguessr.com/maps/bulgaria"),
        ("../countryDataset/train/Cambodia", "https://www.geoguessr.com/maps/cambodia"),
        ("../countryDataset/train/Canada", "https://www.geoguessr.com/maps/canada"),
        ("../countryDataset/train/Chile", "https://www.geoguessr.com/maps/chile"),
        ("../countryDataset/train/Colombia", "https://www.geoguessr.com/maps/colombia"),
        ("../countryDataset/train/Croatia", "https://www.geoguessr.com/maps/croatia"),
        ("../countryDataset/train/Denmark", "https://www.geoguessr.com/maps/denmark"),
        ("../countryDataset/train/DominicanRepublic", "https://www.geoguessr.com/maps/dominican-republic"),
        ("../countryDataset/train/Ecuador", "https://www.geoguessr.com/maps/ecuador"),
        ("../countryDataset/train/Estonia", "https://www.geoguessr.com/maps/estonia"),
        ("../countryDataset/train/Eswatini", "https://www.geoguessr.com/maps/eswatini"),
        ("../countryDataset/train/Finland", "https://www.geoguessr.com/maps/finland"),
        ("../countryDataset/train/France", "https://www.geoguessr.com/maps/france"),
        ("../countryDataset/train/Germany", "https://www.geoguessr.com/maps/germany"),
        ("../countryDataset/train/Ghana", "https://www.geoguessr.com/maps/ghana"),
        ("../countryDataset/train/Greece", "https://www.geoguessr.com/maps/greece"),
        ("../countryDataset/train/Guatemala", "https://www.geoguessr.com/maps/guatemala"),
        ("../countryDataset/train/Hungary", "https://www.geoguessr.com/maps/hungary"),
        ("../countryDataset/train/Iceland", "https://www.geoguessr.com/maps/iceland"),
        ("../countryDataset/train/India", "https://www.geoguessr.com/maps/india"),
        ("../countryDataset/train/Indonesia", "https://www.geoguessr.com/maps/indonesia"),
        ("../countryDataset/train/Ireland", "https://www.geoguessr.com/maps/ireland"),
        ("../countryDataset/train/Israel", "https://www.geoguessr.com/maps/israel"),
        ("../countryDataset/train/Italy", "https://www.geoguessr.com/maps/italy"),
        ("../countryDataset/train/Japan", "https://www.geoguessr.com/maps/japan"),
        ("../countryDataset/train/Jordan", "https://www.geoguessr.com/maps/jordan"),
        ("../countryDataset/train/Kazakhstan", "https://www.geoguessr.com/maps/kazakhstan"),
        ("../countryDataset/train/Kenya", "https://www.geoguessr.com/maps/kenya"),
        ("../countryDataset/train/SouthKorea", "https://www.geoguessr.com/maps/south-korea"),
        ("../countryDataset/train/Kyrgyzstan", "https://www.geoguessr.com/maps/kyrgyzstan"),
        ("../countryDataset/train/Laos", "https://www.geoguessr.com/maps/laos"),
        ("../countryDataset/train/Latvia", "https://www.geoguessr.com/maps/latvia"),
        ("../countryDataset/train/Lebanon", "https://www.geoguessr.com/maps/lebanon"),
        ("../countryDataset/train/Lesotho", "https://www.geoguessr.com/maps/lesotho"),
        ("../countryDataset/train/Lithuania", "https://www.geoguessr.com/maps/lithuania"),
        ("../countryDataset/train/Luxembourg", "https://www.geoguessr.com/maps/luxembourg"),
        ("../countryDataset/train/Madagascar", "https://www.geoguessr.com/maps/madagascar"),
        ("../countryDataset/train/Malaysia", "https://www.geoguessr.com/maps/malaysia"),
        ("../countryDataset/train/Malta", "https://www.geoguessr.com/maps/malta"),
        ("../countryDataset/train/Mexico", "https://www.geoguessr.com/maps/mexico"),
        ("../countryDataset/train/Mongolia", "https://www.geoguessr.com/maps/mongolia"),
        ("../countryDataset/train/Montenegro", "https://www.geoguessr.com/maps/montenegro"),
        ("../countryDataset/train/Netherlands", "https://www.geoguessr.com/maps/netherlands"),
        ("../countryDataset/train/NewZealand", "https://www.geoguessr.com/maps/new-zealand"),
        ("../countryDataset/train/Nigeria", "https://www.geoguessr.com/maps/nigeria"),
        ("../countryDataset/train/NorthMacedonia", "https://www.geoguessr.com/maps/north-macedonia"),
        ("../countryDataset/train/Norway", "https://www.geoguessr.com/maps/norway"),
        ("../countryDataset/train/Oman", "https://www.geoguessr.com/maps/oman"),
        ("../countryDataset/train/Panama", "https://www.geoguessr.com/maps/panama"),
        ("../countryDataset/train/Peru", "https://www.geoguessr.com/maps/peru"),
        ("../countryDataset/train/Philippines", "https://www.geoguessr.com/maps/philippines"),
        ("../countryDataset/train/Poland", "https://www.geoguessr.com/maps/poland"),
        ("../countryDataset/train/Portugal", "https://www.geoguessr.com/maps/portugal"),
        ("../countryDataset/train/Qatar", "https://www.geoguessr.com/maps/qatar"),
        ("../countryDataset/train/Romania", "https://www.geoguessr.com/maps/romania"),
        ("../countryDataset/train/Russia", "https://www.geoguessr.com/maps/russia"),
        ("../countryDataset/train/Rwanda", "https://www.geoguessr.com/maps/rwanda"),
        ("../countryDataset/train/Senegal", "https://www.geoguessr.com/maps/senegal"),
        ("../countryDataset/train/Serbia", "https://www.geoguessr.com/maps/serbia"),
        ("../countryDataset/train/Singapore", "https://www.geoguessr.com/maps/singapore"),
        ("../countryDataset/train/Slovakia", "https://www.geoguessr.com/maps/slovakia"),
        ("../countryDataset/train/Slovenia", "https://www.geoguessr.com/maps/slovenia"),
        ("../countryDataset/train/SouthAfrica", "https://www.geoguessr.com/maps/south-africa"),
        ("../countryDataset/train/Spain", "https://www.geoguessr.com/maps/spain"),
        ("../countryDataset/train/Sweden", "https://www.geoguessr.com/maps/sweden"),
        ("../countryDataset/train/Switzerland", "https://www.geoguessr.com/maps/switzerland"),
        ("../countryDataset/train/Thailand", "https://www.geoguessr.com/maps/thailand"),
        ("../countryDataset/train/Tunisia", "https://www.geoguessr.com/maps/tunisia"),
        ("../countryDataset/train/Turkey", "https://www.geoguessr.com/maps/Turkey"),
        ("../countryDataset/train/Uganda", "https://www.geoguessr.com/maps/uganda"),
        ("../countryDataset/train/Ukraine", "https://www.geoguessr.com/maps/ukraine"),
        ("../countryDataset/train/UnitedArabEmirates", "https://www.geoguessr.com/maps/uae"),
        ("../countryDataset/train/UnitedKingdom", "https://www.geoguessr.com/maps/uk"),
        ("../countryDataset/train/USA", "https://www.geoguessr.com/maps/usa"),
        ("../countryDataset/train/Uruguay", "https://www.geoguessr.com/maps/uruguay"),
    ]

    num_images = 500
    for country, map_url in countries:
        s = time.perf_counter()
        os.makedirs(country, exist_ok=True)

        hashes = get_image_hashes(country)

        for counter in range(1, num_images+1):
            # Prevent map opening again due to mouse hovering on top
            pyautogui.moveTo(MAP_X, CENTER_Y, duration=0.1)
            num_existing_screens, screenshot_path = take_screenshot(country)

            if not is_legal_screenshot(screenshot_path, hashes):
                os.remove(screenshot_path)
                num_existing_screens -= 1

            if num_existing_screens >= num_images:
                break

            simulate_mouse_click(MAP_X, MAP_Y)  # open map
            simulate_mouse_click(MAP_X, MAP_Y)  # Click map
            pyautogui.press("space")  # Make guess
            pyautogui.press("space")  # Next round

            time.sleep(0.05)
            pyautogui.press("space")
            time.sleep(0.05)
            pyautogui.press ("space")

            counter += 1
            # Sometimes GeoGuessr hangs at a screen, refresh once in a while to
            # prevent this
            if counter % 20 == 0:
                pyautogui.hotkey("command", "r")
                time.sleep(6)
                pyautogui.press("space")

            simulate_mouse_click(CENTER_X, CENTER_Y)
            time.sleep(0.3)  # Wait for next round

        time_taken = round(time.perf_counter() - s, 3)
        print(f"Took {time_taken}s")
        pyautogui.hotkey("command", "w")
