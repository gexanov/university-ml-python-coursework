import sys
from pathlib import Path

import cv2
import numpy as np


def select_route_points(mask: np.ndarray, max_points: int = 12, min_distance: int = 35):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    candidates = sorted(zip(xs, ys), key=lambda point: point[1], reverse=True)
    selected = []

    for x, y in candidates:
        too_close = False
        for sx, sy in selected:
            if np.hypot(x - sx, y - sy) < min_distance:
                too_close = True
                break
        if not too_close:
            selected.append((int(x), int(y)))
        if len(selected) >= max_points:
            break

    return sorted(selected, key=lambda point: point[1], reverse=True)


def resize_for_preview(image: np.ndarray, max_width: int = 1400, max_height: int = 900):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def main():
    script_dir = Path(__file__).resolve().parent
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "1.jpg"

    if not image_path.exists():
        print(f"Файл изображения не найден: {image_path}")
        print("Положите изображение фасада в house/facade.jpg или передайте путь аргументом.")
        return

    building = cv2.imread(str(image_path))
    if building is None:
        print(f"Не удалось открыть изображение: {image_path}")
        return

    gray = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris_response = cv2.dilate(harris_response, None)

    _, thresholded = cv2.threshold(
        harris_response,
        0.02 * harris_response.max(),
        255,
        cv2.THRESH_BINARY,
    )
    thresholded = np.uint8(thresholded)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    route_points = select_route_points(thresholded, max_points=15, min_distance=40)
    result = building.copy()

    for index, point in enumerate(route_points, start=1):
        cv2.drawMarker(
            result,
            point,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=40,
            thickness=4,
        )
        cv2.putText(
            result,
            str(index),
            (point[0] + 12, point[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for start, end in zip(route_points, route_points[1:]):
        cv2.line(result, start, end, (0, 255, 0), 4)

    output_path = script_dir / "facade_route_result.jpg"
    preview_path = script_dir / "facade_route_preview.jpg"
    cv2.imwrite(str(output_path), result)

    preview = resize_for_preview(result)
    cv2.imwrite(str(preview_path), preview)

    if route_points:
        print("Список угловых точек маршрута (снизу вверх):")
        for index, (x, y) in enumerate(route_points, start=1):
            print(f"{index}. x={x}, y={y}")
    else:
        print("Подходящие угловые точки не найдены.")

    print(f"Полный результат сохранен в: {output_path}")
    print(f"Уменьшенная версия для просмотра сохранена в: {preview_path}")

    cv2.imshow("Facade Route", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
