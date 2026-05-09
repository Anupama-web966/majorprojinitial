import speech_recognition as sr

# Harassment / emergency words
ALERT_WORDS = [
    "help",
    "save me",
    "leave me",
    "don't touch me",
    "stop it",
    "bitch",
    "bulbul"
]

recognizer = sr.Recognizer()

mic = sr.Microphone()

print("🎤 Voice Detection Started...")

while True:

    with mic as source:

        recognizer.adjust_for_ambient_noise(source)

        print("Listening...")

        audio = recognizer.listen(source)

    try:

        text = recognizer.recognize_google(audio)

        text = text.lower()

        print("You said:", text)

        for word in ALERT_WORDS:

            if word in text:

                print("🚨 HARASSMENT DETECTED:", word)

    except sr.UnknownValueError:
        print("Could not understand")

    except sr.RequestError:
        print("API unavailable")