def trainingModel():
    # Import YOLO dan model pretrained
    from ultralytics import YOLO
    model = YOLO('yolov8m.pt')

    # Import Dataset API dari Roboflow
    # !pip install roboflow

    # from roboflow import Roboflow # Dataset yang sudah ada
    # rf = Roboflow(api_key="aMlgLKvz6zIUJvm4xykm")
    # project = rf.workspace("luke-borozan-wrzn1").project("soccer-detection-v2")
    # dataset = project.version(3).download("yolov8")

    # !pip install roboflow # Dataset hasil labeling dan hanya 200 image (termasuk augmentasi)

    # from roboflow import Roboflow
    # rf = Roboflow(api_key="aMlgLKvz6zIUJvm4xykm")
    # project = rf.workspace("school-zlexb").project("football-scouting")
    # dataset = project.version(1).download("yolov8")

    # !pip install roboflow

    # from roboflow import Roboflow # Dataset hasil labeling dan hanya 911 image segonlyplayer (tidak termasuk augmentasi)
    # rf = Roboflow(api_key="aMlgLKvz6zIUJvm4xykm")
    # project = rf.workspace("school-zlexb").project("football-scouting")
    # dataset = project.version(2).download("yolov8")

    # !pip install roboflow

    # from roboflow import Roboflow # Dataset hasil labeling dan hanya 911 image segplayerball (tidak termasuk augmentasi)
    # rf = Roboflow(api_key="aMlgLKvz6zIUJvm4xykm")
    # project = rf.workspace("school-zlexb").project("football-scouting")
    # dataset = project.version(3).download("yolov8")

    # !pip install roboflow

    # from roboflow import Roboflow # Dataset yang sudah ada
    # rf = Roboflow(api_key="aMlgLKvz6zIUJvm4xykm")
    # project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    # dataset = project.version(1).download("yolov8")

    # from roboflow import Roboflow # Dataset hasil labeling dan hanya 1695 image segonlyplayer (augmentasi flip=horizontal, saturation, brightness)
    # rf = Roboflow(api_key="aMlgLKvz6zIUJvm4xykm")
    # project = rf.workspace("school-zlexb").project("football-scouting")
    # dataset = project.version(4).download("yolov8")

    f = open("Football-Scouting-4/data.yaml", "r")
    print(f.read())

    text = """
    names:
    - ball
    - goalkeeper
    - player
    - referee
    nc: 4
    roboflow:
    license: CC BY 4.0
    project: football-scouting
    url: https://universe.roboflow.com/school-zlexb/football-scouting/dataset/4
    version: 4
    workspace: school-zlexb
    test: C:/Users/maldi/Desktop/TA/Football-Scouting-4/test/images
    train: C:/Users/maldi/Desktop/TA/Football-Scouting-4/train/images
    val: C:/Users/maldi/Desktop/TA/Football-Scouting-4/valid/images
    """

    with open("Football-Scouting-4/data.yaml", "w") as file:
        file.write(text)

    f = open("Football-Scouting-4/data.yaml", "r")
    print(f.read())

    # Train/Fine Tuning dengan dataset yang telah di import
    model.train(data='Football-Scouting-4/data.yaml', imgsz=640, epochs=100)

if __name__ ==  '__main__':
    trainingModel()