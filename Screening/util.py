import os

def latest_file(directory):
    return sorted([os.path.join(directory, file) for file in os.listdir(directory)], key=lambda x: os.path.getmtime(x), reverse=True)[0]