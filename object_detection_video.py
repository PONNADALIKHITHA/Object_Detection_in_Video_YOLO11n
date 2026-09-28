import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_NAME = "yolo11n.pt"
DEFAULT_INPUT_DIR = Path("input_videos")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_RESULTS_DIR = Path("results")


def process_video(
    model, input_path, output_path, show=False, conf=0.25, progress_callback=None
):
    """Run YOLO11n inference continuously on every frame of a video."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )

    frame_count = 0
    detection_count = 0
    class_counts = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO11n inference on the current video frame.
            results = model.predict(source=frame, conf=conf, verbose=False)

            # Post-processing/visualization: draw boxes, labels and confidence.
            annotated = results[0].plot()

            if results[0].boxes is not None:
                boxes = results[0].boxes
                names = results[0].names
                detection_count += len(boxes)

                for cls_id in boxes.cls.tolist():
                    class_name = names[int(cls_id)]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

            writer.write(annotated)

            frame_count += 1
            if progress_callback and (
                frame_count % max(1, int(fps)) == 0 or frame_count == total_frames
            ):
                progress_callback(frame_count, total_frames)

            if show:
                cv2.imshow("YOLO11n - Object Detection in Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    return {
        "input": str(input_path),
        "output": str(output_path),
        "frames_processed": frame_count,
        "total_detections": detection_count,
        "classes_detected": ", ".join(sorted(class_counts.keys())),
    }


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input",
        "output",
        "frames_processed",
        "total_detections",
        "classes_detected",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Object Detection in Video using pre-trained YOLO11n."
    )
    parser.add_argument("--input", type=str, help="Path to one input video.")
    parser.add_argument("--output", type=str, help="Path for one output video.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video while processing. Press Q to stop.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    args = parser.parse_args()

    print("Loading pre-trained YOLO11n...")
    model = YOLO(MODEL_NAME)
    print("YOLO11n loaded successfully.")

    rows = []

    if args.input:
        input_path = Path(args.input)
        output_path = (
            Path(args.output)
            if args.output
            else DEFAULT_OUTPUT_DIR / f"{input_path.stem}_detected.mp4"
        )

        result = process_video(
            model, input_path, output_path, show=args.show, conf=args.conf
        )
        rows.append(result)
        print("\nProcessing completed.")
        print(f"Output: {result['output']}")
        print(f"Frames processed: {result['frames_processed']}")
        print(f"Total detections: {result['total_detections']}")
        print(f"Classes detected: {result['classes_detected']}")

    else:
        videos = sorted(DEFAULT_INPUT_DIR.glob("*.mp4"))
        if not videos:
            print(
                "No videos found. Add test_case_1.mp4, test_case_2.mp4, "
                "and test_case_3.mp4 to input_videos/."
            )
            return

        for index, input_path in enumerate(videos, start=1):
            output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_detected.mp4"
            print(f"\nProcessing test case {index}: {input_path}")
            result = process_video(
                model, input_path, output_path, show=args.show, conf=args.conf
            )
            rows.append(result)
            print(f"Saved: {result['output']}")

    write_summary(rows, DEFAULT_RESULTS_DIR / "detection_summary.csv")
    print("\nDetection summary saved to results/detection_summary.csv")


if __name__ == "__main__":
    main()
