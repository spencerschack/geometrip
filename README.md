# Geometrip

## Usage
`./main.py -f INPUT_FILE -o OUTPUT_FOLDER`

## Example

### This will be our input image:
![Input image](./images/IMG_1153_resized.jpg)

### The image is first blurred:
![Blurred image](./steps/blur.png)

### Then the colors are clustered:
![Clustered colors](./steps/cluster.png)

### A canny filter is applied:
![Canny filter](./steps/canny.png)

### And contours are drawn over the image:
![Contour lines](./steps/contour.png)

### Then a triangle mesh is drawn from the contours:
![Geometry](./steps/geometry.png)

### And the geometry is colored based on the input:
![Final image](./steps/final.png)
