from ultralyticsplus import YOLO, render_result
from PIL import Image

model   = YOLO('keremberke/yolov8m-building-segmentation')
image   = Image.open('../../data/examples/maps/tower_map.jpg')
results = model.predict(image)

print(results[0].boxes)
print(results[0].masks)
render = render_result(model=model, image=image, result=results[0])
render.show()
