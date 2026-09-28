"""
Script to create clean zip archives of the complete solution
"""

import zipfile
import os

def create_zip_archive():
    project_dir = r"C:\Users\Alok Agarwal\.gemini\antigravity-ide\scratch\gradient_descent_mastery"
    zip_dest_1 = os.path.join(project_dir, "gradient_descent_mastery.zip")
    zip_dest_2 = r"D:\gradient_descent\gradient_descent_solution.zip"

    files_to_include = [
        "app.py",
        "gradient_descent_pipeline.ipynb",
        "submission.csv",
        "requirements.txt",
        "README.md",
        "generate_submission.py",
        "build_notebook.py"
    ]

    src_files = [
        os.path.join("src", f) for f in os.listdir(os.path.join(project_dir, "src")) if f.endswith(".py")
    ]

    all_files = files_to_include + src_files

    for dest in [zip_dest_1, zip_dest_2]:
        print(f"Creating zip file at: {dest}")
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for rel_path in all_files:
                abs_path = os.path.join(project_dir, rel_path)
                if os.path.exists(abs_path):
                    z.write(abs_path, arcname=os.path.join("gradient_descent_mastery", rel_path))
                    print(f"  + Added {rel_path} ({os.path.getsize(abs_path):,} bytes)")
                else:
                    print(f"  - Warning: Missing {abs_path}")
        print(f"Finished {dest}. Total Zip Size: {os.path.getsize(dest):,} bytes\n")

if __name__ == "__main__":
    create_zip_archive()
