# Home Security for UCAM24/7Live Cameras

Add to `.env` file:
```
CAMERA_USERNAME=your_camera_username
CAMERA_PASSWORD=your_camera_password
IP_ADDRESS=your_camera_ip_address
```

Run the following command to start the program:
```
python3 observe_frames.py
```

This save a frame every time a person is detected in the `imgs/` folder.
