from jetson.inference import detectNet
from jetson.utils import videoSource.videoOutput
net = detectNet("ssd-mobilenet-v2"，threshold=0.5)
camera = videoSource("/home/nvidia/jetson-inference/data/images/humans_7.jpg")
display = videoOutput("display://0")
while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        continue
    detections = net.Detect(img)
    print(detections[0])

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
