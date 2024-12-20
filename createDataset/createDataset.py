import os
import time
import pyautogui
from PIL import ImageGrab

MAP_X, MAP_Y = 1779, -183
GUESS_X, GUESS_Y = 1789, -40
CENTER_X, CENTER_Y = 801, -634


def take_screenshot(directory):
    i = 1
    while os.path.exists(os.path.join(directory, f"{i}.png")):
        i += 1

    screenshot = ImageGrab.grab(bbox=(0, -1000, 1600, -100))
    screenshot_path = os.path.join(directory, f"{i}.png")
    screenshot.save(screenshot_path)

    print(f"Screenshot saved to {screenshot_path}")

    return i


def simulate_mouse_click(x, y):
    try:
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()
    except pyautogui.FailSafeError:
        print("Mouse moved to a fail-safe corner. Click cancelled.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # states = [
        # ("Iowa", "https://www.geoguessr.com/maps/62c71ff04b6aed0f2717be4e"),
        # ("Kansas", "https://www.geoguessr.com/maps/62f7c3f8e0044554ce4be286"),
        # ("Kentucky", "https://www.geoguessr.com/maps/5e12e7d62034a03444032b7c"),
        # ("Louisiana", "https://www.geoguessr.com/maps/62881aac8119d0fec2fa9833"),
        # ("Maine", "https://www.geoguessr.com/maps/62d1d918f271371fa80eebfb"),
        # ("Maryland", "https://www.geoguessr.com/maps/6024a0565c13a9000137ae50"),
        # ("Massachusetts", "https://www.geoguessr.com/maps/6240fca087300c4d18cad7ce"),
        # ("Michigan", "https://www.geoguessr.com/maps/62318d9b9af9e10001eef28d"),
        # ("Minnesota", "https://www.geoguessr.com/maps/5b06f033faa4cf3ce43af565"),
        # ("Mississippi", "https://www.geoguessr.com/maps/619e97d154b03e000105ece2"),
        # ("Missouri", "https://www.geoguessr.com/maps/5b2452ca7a2b425ef47915dc"),
        # ("Montana", "https://www.geoguessr.com/maps/630008c0ee565627ad61eea5"),
        # ("Nebraska", "https://www.geoguessr.com/maps/62db3f3f7442e56b4bae4a22"),
        # ("Nevada", "https://www.geoguessr.com/maps/62fc7ffc7d19504f1044b817"),
        # ("NewHampshire", "https://www.geoguessr.com/maps/601deb05aad32d0001b46698"),
        # ("NewJersey", "https://www.geoguessr.com/maps/6611bc3eb53d4b28f6b8481d"),
        # ("NewMexico", "https://www.geoguessr.com/maps/62fce8c91a5a73c965780202"),
        # ("NewYork", "https://www.geoguessr.com/maps/622e2da932c26a0001d7e159"),
        # ("NorthCarolina", "https://www.geoguessr.com/maps/5b16061a4559f41ad858b39c"),
        # ("NorthDakota", "https://www.geoguessr.com/maps/62fce964c1119d3d367eaa97"),
        # ("Ohio", "https://www.geoguessr.com/maps/65b4c964c05ed7c170b8419a"),
        # ("Oklahoma", "https://www.geoguessr.com/maps/629c26f9f746380322937585"),
        # ("Oregon", "https://www.geoguessr.com/maps/5af31213582de22a00dfd2de"),
        # ("Pennsylvania", "https://www.geoguessr.com/maps/614cab360d99a40001c48733"),
        # ("RhodeIsland", "https://www.geoguessr.com/maps/624db7cd46484446a435d66c"),
        # ("SouthCarolina", "https://www.geoguessr.com/maps/628e5e0a2564e4c431d457c7"),
        # ("SouthDakota", "https://www.geoguessr.com/maps/6306cb538a77fe6f90866088"),
        # ("Tennessee", "https://www.geoguessr.com/maps/62eb5aaa58331a9559134f7a"),
        # ("Texas", "https://www.geoguessr.com/maps/64d2d37cd170f9e091f1cd5b"),
        # ("Utah", "https://www.geoguessr.com/maps/62e98752c8455c2b59e24e05"),
        # ("Vermont", "https://www.geoguessr.com/maps/5b247799cead176c38290345"),
        # ("Virginia", "https://www.geoguessr.com/maps/644dda220a5436bd83195afb"),
        # ("Washington", "https://www.geoguessr.com/maps/62fe4f7c081b3f07d0c5ec84"),
        # ("WashingtonDC", "https://www.geoguessr.com/maps/56f9769d148a781b143b268c"),
        # ("WestVirginia", "https://www.geoguessr.com/maps/60cfcbc47abe710001f7d2bf"),
        # ("Wisconsin", "https://www.geoguessr.com/maps/5b247b8faf2b2a3cd8a1c792"),
        # ("Wyoming", "https://www.geoguessr.com/maps/62fd6975a425c1c7c196c5fe"),
    # ]
    countries = [
        ("../countryDataset/Albania", "https://www.geoguessr.com/maps/albania"),
        ("../countryDataset/Argentina", "https://www.geoguessr.com/maps/argentina"),
        ("../countryDataset/Australia", "https://www.geoguessr.com/maps/australia"),
        ("../countryDataset/Austria", "https://www.geoguessr.com/maps/austria"),
        ("../countryDataset/Bangladesh", "https://www.geoguessr.com/maps/bangladesh"),
        ("../countryDataset/Belgium", "https://www.geoguessr.com/maps/belgium"),
        ("../countryDataset/Bhutan", "https://www.geoguessr.com/maps/bhutan"),
        ("../countryDataset/Bolivia", "https://www.geoguessr.com/maps/bolivia"),
        ("../countryDataset/Botswana", "https://www.geoguessr.com/maps/botswana"),
        ("../countryDataset/Brazil", "https://www.geoguessr.com/maps/brazil"),
        ("../countryDataset/Bulgaria", "https://www.geoguessr.com/maps/bulgaria"),
        ("../countryDataset/Cambodia", "https://www.geoguessr.com/maps/cambodia"),
        ("../countryDataset/Canada", "https://www.geoguessr.com/maps/canada"),
        ("../countryDataset/Chile", "https://www.geoguessr.com/maps/chile"),
        ("../countryDataset/Colombia", "https://www.geoguessr.com/maps/colombia"),
        ("../countryDataset/Croatia", "https://www.geoguessr.com/maps/croatia"),
        ("../countryDataset/Denmark", "https://www.geoguessr.com/maps/denmark"),
        ("../countryDataset/DominicanRepublic", "https://www.geoguessr.com/maps/dominican-republic"),
        ("../countryDataset/Ecuador", "https://www.geoguessr.com/maps/ecuador"),
        ("../countryDataset/Estonia", "https://www.geoguessr.com/maps/estonia"),
        ("../countryDataset/Eswatini", "https://www.geoguessr.com/maps/eswatini"),
        ("../countryDataset/Finland", "https://www.geoguessr.com/maps/finland"),
        ("../countryDataset/France", "https://www.geoguessr.com/maps/france"),
        ("../countryDataset/Germany", "https://www.geoguessr.com/maps/germany"),
        ("../countryDataset/Ghana", "https://www.geoguessr.com/maps/ghana"),
        ("../countryDataset/Greece", "https://www.geoguessr.com/maps/greece"),
        ("../countryDataset/Guatemala", "https://www.geoguessr.com/maps/guatemala"),
        ("../countryDataset/Hungary", "https://www.geoguessr.com/maps/hungary"),
        ("../countryDataset/Iceland", "https://www.geoguessr.com/maps/iceland"),
        ("../countryDataset/India", "https://www.geoguessr.com/maps/india"),
        ("../countryDataset/Indonesia", "https://www.geoguessr.com/maps/indonesia"),
        ("../countryDataset/Ireland", "https://www.geoguessr.com/maps/ireland"),
        ("../countryDataset/Israel", "https://www.geoguessr.com/maps/israel"),
        ("../countryDataset/Italy", "https://www.geoguessr.com/maps/italy"),
        ("../countryDataset/Japan", "https://www.geoguessr.com/maps/japan"),
        ("../countryDataset/Jordan", "https://www.geoguessr.com/maps/jordan"),
        ("../countryDataset/Kazakhstan", "https://www.geoguessr.com/maps/kazakhstan"),
        ("../countryDataset/Kenya", "https://www.geoguessr.com/maps/kenya"),
        ("../countryDataset/SouthKorea", "https://www.geoguessr.com/maps/south-korea"),
        ("../countryDataset/Kyrgyzstan", "https://www.geoguessr.com/maps/kyrgyzstan"),
        ("../countryDataset/Laos", "https://www.geoguessr.com/maps/laos"),
        ("../countryDataset/Latvia", "https://www.geoguessr.com/maps/latvia"),
        ("../countryDataset/Lebanon", "https://www.geoguessr.com/maps/lebanon"),
        ("../countryDataset/Lesotho", "https://www.geoguessr.com/maps/lesotho"),
        ("../countryDataset/Lithuania", "https://www.geoguessr.com/maps/lithuania"),
        ("../countryDataset/Luxembourg", "https://www.geoguessr.com/maps/luxembourg"),
        ("../countryDataset/Madagascar", "https://www.geoguessr.com/maps/madagascar"),
        ("../countryDataset/Malaysia", "https://www.geoguessr.com/maps/malaysia"),
        ("../countryDataset/Malta", "https://www.geoguessr.com/maps/malta"),
        ("../countryDataset/Mexico", "https://www.geoguessr.com/maps/mexico"),
        ("../countryDataset/Mongolia", "https://www.geoguessr.com/maps/mongolia"),
        ("../countryDataset/Montenegro", "https://www.geoguessr.com/maps/montenegro"),
        ("../countryDataset/Netherlands", "https://www.geoguessr.com/maps/netherlands"),
        ("../countryDataset/NewZealand", "https://www.geoguessr.com/maps/new-zealand"),
        ("../countryDataset/Nigeria", "https://www.geoguessr.com/maps/nigeria"),
        ("../countryDataset/NorthMacedonia", "https://www.geoguessr.com/maps/north-macedonia"),
        ("../countryDataset/Norway", "https://www.geoguessr.com/maps/norway"),
        ("../countryDataset/Oman", "https://www.geoguessr.com/maps/oman"),
        ("../countryDataset/Panama", "https://www.geoguessr.com/maps/panama"),
        ("../countryDataset/Peru", "https://www.geoguessr.com/maps/peru"),
        ("../countryDataset/Philippines", "https://www.geoguessr.com/maps/philippines"),
        ("../countryDataset/Poland", "https://www.geoguessr.com/maps/poland"),
        ("../countryDataset/Portugal", "https://www.geoguessr.com/maps/portugal"),
        ("../countryDataset/Qatar", "https://www.geoguessr.com/maps/qatar"),
        ("../countryDataset/Romania", "https://www.geoguessr.com/maps/romania"),
        ("../countryDataset/Russia", "https://www.geoguessr.com/maps/russia"),
        ("../countryDataset/Rwanda", "https://www.geoguessr.com/maps/rwanda"),
        ("../countryDataset/Senegal", "https://www.geoguessr.com/maps/senegal"),
        ("../countryDataset/Serbia", "https://www.geoguessr.com/maps/serbia"),
        ("../countryDataset/Singapore", "https://www.geoguessr.com/maps/singapore"),
        ("../countryDataset/Slovakia", "https://www.geoguessr.com/maps/slovakia"),
        ("../countryDataset/Slovenia", "https://www.geoguessr.com/maps/slovenia"),
        ("../countryDataset/SouthAfrica", "https://www.geoguessr.com/maps/south-africa"),
        ("../countryDataset/Spain", "https://www.geoguessr.com/maps/spain"),
        ("../countryDataset/Sweden", "https://www.geoguessr.com/maps/sweden"),
        ("../countryDataset/Switzerland", "https://www.geoguessr.com/maps/switzerland"),
        ("../countryDataset/Thailand", "https://www.geoguessr.com/maps/thailand"),
        ("../countryDataset/Tunisia", "https://www.geoguessr.com/maps/tunisia"),
        ("../countryDataset/Turkey", "https://www.geoguessr.com/maps/Turkey"),
        ("../countryDataset/Uganda", "https://www.geoguessr.com/maps/uganda"),
        ("../countryDataset/Ukraine", "https://www.geoguessr.com/maps/ukraine"),
        ("../countryDataset/UnitedArabEmirates", "https://www.geoguessr.com/maps/uae"),
        ("../countryDataset/UnitedKingdom", "https://www.geoguessr.com/maps/uk"),
        ("../countryDataset/USA", "https://www.geoguessr.com/maps/usa"),
        ("../countryDataset/Uruguay", "https://www.geoguessr.com/maps/uruguay"),
    ]

    num_images = 310
    for country, map_url in countries:
        s = time.perf_counter()
        os.makedirs(country, exist_ok=True)


        for counter in range(1, num_images+1):
            pyautogui.moveTo(MAP_X, CENTER_Y, duration=0.1)  # Prevent map opening again
            num_existing_screens = take_screenshot(country)
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
            time.sleep(0.9)  # Wait for next round

        time_taken = round(time.perf_counter() - s, 3)
        print(f"Took {time_taken}s")
        pyautogui.hotkey("command", "w")
