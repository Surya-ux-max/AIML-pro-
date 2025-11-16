import subprocess
import webbrowser
import time
import threading

def open_browser():
    time.sleep(2)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("Starting Smart City Prediction System...")
    print("Web interface will open at: http://127.0.0.1:5000")
    
    # Start browser in separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask app
    subprocess.run(['python', 'app.py'])