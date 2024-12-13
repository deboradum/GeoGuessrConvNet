import re
import os
import hashlib
from PIL import Image


def find_and_delete_duplicates(directory, extensions=None, delete=False):
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    def file_hash(filepath):
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    hashes = {}
    duplicates = []

    for root, _, files in os.walk(directory):
        files.sort()
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                try:
                    hash_value = file_hash(filepath)
                    if hash_value in hashes:
                        duplicates.append(filepath)
                        if delete:
                            os.remove(filepath)
                            print(f"Deleted duplicate: {filepath}")
                        else:
                            print(f"Found duplicate: {filepath}")
                    else:
                        hashes[hash_value] = filepath
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")

    return duplicates


def delete_files(directory, threshold=300, dry_run=True):
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    # Match '1.png', '2.png', etc.
    numeric_file_pattern = re.compile(r"^(\d+)\.png$")

    files_to_delete = []

    for file_name in os.listdir(directory):
        match = numeric_file_pattern.match(file_name)
        if match:
            file_number = int(match.group(1))
            if file_number > threshold:
                files_to_delete.append(file_name)

    if dry_run:
        print("Dry run: The following files would be deleted:")
        for file in files_to_delete:
            print(file)
    else:
        for file in files_to_delete:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
        print("The following files were deleted:")
        for file in files_to_delete:
            print(file)

    return files_to_delete


def check_valid(directory):
    for file_name in os.listdir(directory):
        if file_name == ".DS_Store":
            continue
        try:
            Image.open(f"{directory}/{file_name}")
        except Exception as e:
            print(e)


locations = [
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

for country, map_url in locations:
    find_and_delete_duplicates(country, extensions=[".png"], delete=True)
    check_valid(country)
    delete_files(country, threshold=300, dry_run=True)
