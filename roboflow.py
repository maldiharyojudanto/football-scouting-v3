from roboflow import Roboflow
rf = Roboflow(api_key="aMlgLKvz6zIUJvm4xykm")
project = rf.workspace("school-zlexb").project("football-scouting")
dataset = project.version(5).download("yolov8")
