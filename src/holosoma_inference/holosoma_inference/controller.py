import time
from holosoma_inference.utils.move_command import move

# Drive forward for 3 seconds
move(vx=0.5, vy=0.0, vyaw=0.0)
time.sleep(3)

# Stop
move(vx=0.0, vy=0.0, vyaw=0.0)