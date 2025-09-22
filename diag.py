# diag.py
# This script is for diagnostic purposes.
# It attempts to import the FastAPI app to reveal any import-time errors.

print("Attempting to import the application...")
try:
    from main import app
    print("Application imported successfully!")
except Exception as e:
    print("An error occurred during import:")
    print(e)
    import traceback
    traceback.print_exc()
