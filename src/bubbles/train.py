import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO models")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["nano", "small"],
        required=True,
        help="YOLO model size to train"
        )
    args = parser.parse_args()
    print(f"Model: {args.model}")
    models = {
        "nano": "yolo26n-sem.pt",
        "small": "yolo26s-sem.pt"
    }
    model = YOLO(models[args.model])
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        patience=15,
        seed=42,
        name=f"experiment_cloud_{args.model}"
    )
if __name__ == "__main__":
    main()