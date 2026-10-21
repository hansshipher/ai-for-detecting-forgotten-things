from pathlib import Path
from ultralytics import YOLO
import config as cfg

def main():
    print(f"Model download")
    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Save path: {cfg.MODEL_PATH}")

    cfg.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Folder created/checked: {cfg.MODEL_PATH.parent}")
    
    if cfg.MODEL_PATH.exists():
        print(f"Model already exists: {cfg.MODEL_PATH}")
        return
    
    print("Starting download...")
    model = YOLO(str(cfg.MODEL_PATH))
    
    if cfg.MODEL_PATH.exists():
        size_mb = cfg.MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Model is successfully downloaded!")
        print(f"  Path: {cfg.MODEL_PATH}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print("Error: file not saved")

if __name__ == "__main__":
    main()