from ultralytics import NAS

# Load YOLO-NAS-S model
model = NAS('yolo_nas_s.pt')  # You can specify other variants like 'yolonas-m', etc.
model.save('yolonas_s.pt')

model.train(data='data.yaml', epochs=50, batch=16, imgsz=640)
model.save('yolonas_s_finetuned.pt')

results = model.val()

# Test on a custom image
#img = 'path/to/your/image.jpg'
#predictions = model(img)
#predictions.show()  # Display the results

model.export(format='onnx')