import os
import zipfile

main_folder = "/raid/nlp/aravind"   # change this
zip_name = "python_files_only.zip"

with zipfile.ZipFile(zip_name, 'w') as zipf:
    for filename in os.listdir(main_folder):
        if filename.endswith(".py") and not filename.startswith('.'):
            filepath = os.path.join(main_folder, filename)
            zipf.write(filepath, arcname=filename)
