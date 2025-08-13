# File: main.py
# Updated: 2025-08-13
"""
Main launcher for the Image Captioning + Segmentation project.
Runs the Streamlit app with correct settings.
"""

import os
import sys
import subprocess
import webbrowser
import platform

def open_outputs_folder():
    """Open the outputs folder if running locally (not Codespaces)."""
    outputs_path = os.path.join(os.path.dirname(__file__), "outputs")
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    
    try:
        if "CODESPACES" not in os.environ:
            if platform.system() == "Windows":
                os.startfile(outputs_path)  # type: ignore
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", outputs_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", outputs_path])
    except Exception as e:
        print(f"⚠ Could not open outputs folder: {e}")

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "app", "streamlit_app.py")

    if not os.path.exists(app_path):
        print(f"❌ ERROR: Streamlit app not found at {app_path}")
        sys.exit(1)

    print("\n🚀 Starting Streamlit app...")
    print("📌 Usage:")
    print("   Local:     Open http://localhost:8501 after running this")
    print("   Codespaces:")
    print("      1. Make port 8501 Public in Ports tab")
    print("      2. Open the provided https://<codespace-id>-8501.app.github.dev link\n")

    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    subprocess.run(cmd)

    # After app closes, open outputs folder
    open_outputs_folder()

if __name__ == "__main__":
    main()
