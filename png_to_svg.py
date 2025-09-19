import cv2
import numpy as np
from pathlib import Path
import svgwrite
from PIL import Image

def raster_to_svg_cv(input_path: str, output_path: str, threshold: int = 128, 
                    fill_color: str = "black", invert: bool = False):
    """
    Convert a PNG/JPG image to SVG using OpenCV contours (no potrace).
    """
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load image in grayscale
        img = Image.open(input_file).convert("L")
        img_array = np.array(img)

        # Threshold to black & white
        if invert:
            _, bw = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, bw = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create SVG canvas
        dwg = svgwrite.Drawing(output_path, profile="tiny",
                                size=(f"{img.width}px", f"{img.height}px"))
        
        for contour in contours:
            # Convert contour to path string
            path_data = []
            for i, point in enumerate(contour):
                x, y = point[0]
                if i == 0:
                    path_data.append(f"M {x} {y}")
                else:
                    path_data.append(f"L {x} {y}")
            path_data.append("Z")  # Close path

            dwg.add(dwg.path(d=" ".join(path_data), fill=fill_color, stroke="none"))

        # Save SVG
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dwg.save()

        print(f"SVG saved at {output_path}, traced {len(contours)} contours.")
        return output_path

    except Exception as e:
        print(f"Error: {e}")
        raise


def batch_raster_to_svg_cv(input_dir: str, output_dir: str, **kwargs):
    """
    Convert all PNG/JPG files in a directory to SVG format (OpenCV version).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))

    if not image_files:
        print(f"No PNG/JPG files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to convert")

    for img_file in image_files:
        svg_file = output_path / f"{img_file.stem}.svg"
        try:
            raster_to_svg_cv(str(img_file), str(svg_file), **kwargs)
        except Exception as e:
            print(f"Failed to convert {img_file.name}: {e}")


# Example usage
if __name__ == "__main__":
    # Single file
    raster_to_svg_cv("images.jpeg", "output_cv.svg", threshold=120, fill_color="navy")

    # Batch
    # batch_raster_to_svg_cv("input_images/", "output_svgs/", threshold=100, invert=True)
