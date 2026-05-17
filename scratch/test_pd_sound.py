import socket
import time

def test_pd():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 30025))
        print("Connected to PD")
        
        # Send READY_ON bang
        sock.sendall(b"READY_ON bang;")
        print("Sent READY_ON")
        time.sleep(0.5)
        
        # Send some values to make sure it's alive
        sock.sendall(b"V 1.0; A 0.5; D 1.0; W 0.0; F 0.0;")
        print("Sent V A D W F")
        
        sock.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pd()
