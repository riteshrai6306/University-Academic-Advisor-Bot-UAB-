import os
import shutil

vectorstore_dir = "vectorstore"
if os.path.exists(vectorstore_dir):
    shutil.rmtree(vectorstore_dir)
    print("Deleted vectorstore directory")
else:
    print("Vectorstore directory not found")