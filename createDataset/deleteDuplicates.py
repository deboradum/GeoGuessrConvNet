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


def delete_files(directory, start_threshold=0, end_threshold=300, dry_run=True):
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    # Match '1.png', '2.png', etc.
    numeric_file_pattern = re.compile(r"^(\d+)\.png$")

    files_to_delete = []

    for file_name in os.listdir(directory):
        match = numeric_file_pattern.match(file_name)
        if match:
            file_number = int(match.group(1))
            if file_number < start_threshold or file_number > end_threshold:
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
    ("../countryDataset/test/Albania", "https://www.geoguessr.com/maps/albania"),
    ("../countryDataset/test/Argentina", "https://www.geoguessr.com/maps/argentina"),
    ("../countryDataset/test/Australia", "https://www.geoguessr.com/maps/australia"),
    ("../countryDataset/test/Austria", "https://www.geoguessr.com/maps/austria"),
    ("../countryDataset/test/Bangladesh", "https://www.geoguessr.com/maps/bangladesh"),
    ("../countryDataset/test/Belgium", "https://www.geoguessr.com/maps/belgium"),
    ("../countryDataset/test/Bhutan", "https://www.geoguessr.com/maps/bhutan"),
    ("../countryDataset/test/Bolivia", "https://www.geoguessr.com/maps/bolivia"),
    ("../countryDataset/test/Botswana", "https://www.geoguessr.com/maps/botswana"),
    ("../countryDataset/test/Brazil", "https://www.geoguessr.com/maps/brazil"),
    ("../countryDataset/test/Bulgaria", "https://www.geoguessr.com/maps/bulgaria"),
    ("../countryDataset/test/Cambodia", "https://www.geoguessr.com/maps/cambodia"),
    ("../countryDataset/test/Canada", "https://www.geoguessr.com/maps/canada"),
    ("../countryDataset/test/Chile", "https://www.geoguessr.com/maps/chile"),
    ("../countryDataset/test/Colombia", "https://www.geoguessr.com/maps/colombia"),
    ("../countryDataset/test/Croatia", "https://www.geoguessr.com/maps/croatia"),
    ("../countryDataset/test/Denmark", "https://www.geoguessr.com/maps/denmark"),
    ("../countryDataset/test/DominicanRepublic", "https://www.geoguessr.com/maps/dominican-republic"),
    ("../countryDataset/test/Ecuador", "https://www.geoguessr.com/maps/ecuador"),
    ("../countryDataset/test/Estonia", "https://www.geoguessr.com/maps/estonia"),
    ("../countryDataset/test/Eswatini", "https://www.geoguessr.com/maps/eswatini"),
    ("../countryDataset/test/Finland", "https://www.geoguessr.com/maps/finland"),
    ("../countryDataset/test/France", "https://www.geoguessr.com/maps/france"),
    ("../countryDataset/test/Germany", "https://www.geoguessr.com/maps/germany"),
    ("../countryDataset/test/Ghana", "https://www.geoguessr.com/maps/ghana"),
    ("../countryDataset/test/Greece", "https://www.geoguessr.com/maps/greece"),
    ("../countryDataset/test/Guatemala", "https://www.geoguessr.com/maps/guatemala"),
    ("../countryDataset/test/Hungary", "https://www.geoguessr.com/maps/hungary"),
    ("../countryDataset/test/Iceland", "https://www.geoguessr.com/maps/iceland"),
    ("../countryDataset/test/India", "https://www.geoguessr.com/maps/india"),
    ("../countryDataset/test/Indonesia", "https://www.geoguessr.com/maps/indonesia"),
    ("../countryDataset/test/Ireland", "https://www.geoguessr.com/maps/ireland"),
    ("../countryDataset/test/Israel", "https://www.geoguessr.com/maps/israel"),
    ("../countryDataset/test/Italy", "https://www.geoguessr.com/maps/italy"),
    ("../countryDataset/test/Japan", "https://www.geoguessr.com/maps/japan"),
    ("../countryDataset/test/Jordan", "https://www.geoguessr.com/maps/jordan"),
    ("../countryDataset/test/Kazakhstan", "https://www.geoguessr.com/maps/kazakhstan"),
    ("../countryDataset/test/Kenya", "https://www.geoguessr.com/maps/kenya"),
    ("../countryDataset/test/SouthKorea", "https://www.geoguessr.com/maps/south-korea"),
    ("../countryDataset/test/Kyrgyzstan", "https://www.geoguessr.com/maps/kyrgyzstan"),
    ("../countryDataset/test/Laos", "https://www.geoguessr.com/maps/laos"),
    ("../countryDataset/test/Latvia", "https://www.geoguessr.com/maps/latvia"),
    ("../countryDataset/test/Lebanon", "https://www.geoguessr.com/maps/lebanon"),
    ("../countryDataset/test/Lesotho", "https://www.geoguessr.com/maps/lesotho"),
    ("../countryDataset/test/Lithuania", "https://www.geoguessr.com/maps/lithuania"),
    ("../countryDataset/test/Luxembourg", "https://www.geoguessr.com/maps/luxembourg"),
    ("../countryDataset/test/Madagascar", "https://www.geoguessr.com/maps/madagascar"),
    ("../countryDataset/test/Malaysia", "https://www.geoguessr.com/maps/malaysia"),
    ("../countryDataset/test/Malta", "https://www.geoguessr.com/maps/malta"),
    ("../countryDataset/test/Mexico", "https://www.geoguessr.com/maps/mexico"),
    ("../countryDataset/test/Mongolia", "https://www.geoguessr.com/maps/mongolia"),
    ("../countryDataset/test/Montenegro", "https://www.geoguessr.com/maps/montenegro"),
    ("../countryDataset/test/Netherlands", "https://www.geoguessr.com/maps/netherlands"),
    ("../countryDataset/test/NewZealand", "https://www.geoguessr.com/maps/new-zealand"),
    ("../countryDataset/test/Nigeria", "https://www.geoguessr.com/maps/nigeria"),
    ("../countryDataset/test/NorthMacedonia", "https://www.geoguessr.com/maps/north-macedonia"),
    ("../countryDataset/test/Norway", "https://www.geoguessr.com/maps/norway"),
    ("../countryDataset/test/Oman", "https://www.geoguessr.com/maps/oman"),
    ("../countryDataset/test/Panama", "https://www.geoguessr.com/maps/panama"),
    ("../countryDataset/test/Peru", "https://www.geoguessr.com/maps/peru"),
    ("../countryDataset/test/Philippines", "https://www.geoguessr.com/maps/philippines"),
    ("../countryDataset/test/Poland", "https://www.geoguessr.com/maps/poland"),
    ("../countryDataset/test/Portugal", "https://www.geoguessr.com/maps/portugal"),
    ("../countryDataset/test/Qatar", "https://www.geoguessr.com/maps/qatar"),
    ("../countryDataset/test/Romania", "https://www.geoguessr.com/maps/romania"),
    ("../countryDataset/test/Russia", "https://www.geoguessr.com/maps/russia"),
    ("../countryDataset/test/Rwanda", "https://www.geoguessr.com/maps/rwanda"),
    ("../countryDataset/test/Senegal", "https://www.geoguessr.com/maps/senegal"),
    ("../countryDataset/test/Serbia", "https://www.geoguessr.com/maps/serbia"),
    ("../countryDataset/test/Singapore", "https://www.geoguessr.com/maps/singapore"),
    ("../countryDataset/test/Slovakia", "https://www.geoguessr.com/maps/slovakia"),
    ("../countryDataset/test/Slovenia", "https://www.geoguessr.com/maps/slovenia"),
    ("../countryDataset/test/SouthAfrica", "https://www.geoguessr.com/maps/south-africa"),
    ("../countryDataset/test/Spain", "https://www.geoguessr.com/maps/spain"),
    ("../countryDataset/test/Sweden", "https://www.geoguessr.com/maps/sweden"),
    ("../countryDataset/test/Switzerland", "https://www.geoguessr.com/maps/switzerland"),
    ("../countryDataset/test/Thailand", "https://www.geoguessr.com/maps/thailand"),
    ("../countryDataset/test/Tunisia", "https://www.geoguessr.com/maps/tunisia"),
    ("../countryDataset/test/Turkey", "https://www.geoguessr.com/maps/Turkey"),
    ("../countryDataset/test/Uganda", "https://www.geoguessr.com/maps/uganda"),
    ("../countryDataset/test/Ukraine", "https://www.geoguessr.com/maps/ukraine"),
    ("../countryDataset/test/UnitedArabEmirates", "https://www.geoguessr.com/maps/uae"),
    ("../countryDataset/test/UnitedKingdom", "https://www.geoguessr.com/maps/uk"),
    ("../countryDataset/test/USA", "https://www.geoguessr.com/maps/usa"),
    ("../countryDataset/test/Uruguay", "https://www.geoguessr.com/maps/uruguay"),
]

for country, map_url in locations:
    # find_and_delete_duplicates(country, extensions=[".png"], delete=True)
    # check_valid(country)
    delete_files(country, start_threshold=241, end_threshold=300, dry_run=False)
