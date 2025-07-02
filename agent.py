import cv2
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import os

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

class FaceAgent(Agent):
    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)
        self.position = position
        self.distance_to_camera = None

    def step(self):
        for other_agent in self.model.schedule.agents:
            if other_agent != self:
                self.influence_other_agent(other_agent)

    def influence_other_agent(self, other_agent):
        x, y = self.position
        other_x, other_y = other_agent.position
        delta_x = (other_x - x) * 0.02
        delta_y = (other_y - y) * 0.02
        self.model.space.move_agent(self, (x + delta_x, y + delta_y))

def detect_faces(image):
    facecas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facecas.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def print_distances(frame, faces, agents, model):
    for agent, (x, y, w, h) in zip(agents, faces):
        a= x+w // 2
        b=y+h // 2
        agent.position = (a,b)
        agent.position2=(a+30,b+50)
        agent.distance_to_camera = model.calculate_distance(max(w, h))

        color = (0, 255, 0)

        if agent.position[1] < 290:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        distance_text = f"Dist: {agent.distance_to_camera:.2f} units"
        textS = cv2.getTextSize(distance_text, font, 0.5, 1)[0]
        textX = x + (w - textS[0]) // 2
        textY = y + h + 15
        cv2.putText(frame, distance_text, (textX, textY), font, 0.5, color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"Agent 0: {agents[0].position}", (10, frame.shape[0] - 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Agent 1: {agent.position2}", (frame.shape[1] // 2 + 10, frame.shape[0] - 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

class FaceDistanceModel(Model):
    def __init__(self, num_agents, reference_distance=100):
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            agent_reporters={"Position": "position", "Distance": "distance_to_camera"}
        )
        self.reference_distance = reference_distance

        for i in range(num_agents):
            agent = FaceAgent(i, self, (0, 0))
            self.schedule.add(agent)

    def calculate_distance(self, face_size):
        focal_length = 500.0
        return (self.reference_distance * focal_length) / face_size

    def step(self):
        self.datacollector.collect(self)
        for agent in self.schedule.agents:
            agent.step()

cascade_path = r'C:/Users/Pc/Documents/hass.xml'
haar_cas = cv2.CascadeClassifier(cascade_path)

p = []
for i in os.listdir(r'C:/Users/Pc/Desktop/padugeevitham'):
    p.append(i)

facereco = cv2.face.LBPHFaceRecognizer_create()
facereco.read('face_trained5_0.yml')

video_capture = cv2.VideoCapture(0)

num_agents = 2
model = FaceDistanceModel(num_agents)

while True:
    ret, frame = video_capture.read()
    faces = detect_faces(frame)
    img = rescaleFrame(frame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('person', gray)

    faces_recog = haar_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for agent, (x, y, w, h) in zip(model.schedule.agents, faces):
        agent.position = (x + w // 2, y + h // 2)
        for (x, y, w, h) in faces_recog:
            faces_roi = gray[y:y + h, x:x + h]
            label, confidence = facereco.predict(faces_roi)
            cv2.putText(frame, p[int(label)], (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)

    print_distances(frame, faces, model.schedule.agents, model)
    
    cv2.imshow('Detect face', frame)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

video_capture.release()
cv2.destroyAllWindows()
