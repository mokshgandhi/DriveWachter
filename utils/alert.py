import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 160)

def alert_user(msg="Pothole ahead!"):
    print(msg)
    engine.say(msg)
    engine.runAndWait()
