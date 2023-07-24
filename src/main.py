import gym
import cv2
import numpy as np
env = gym.make('InvertedPendulum-v4',render_mode="rgb_array")
env.reset()

while True:
    action = env.action_space.sample()  # Replace this with your chosen action selection logic
    observation, reward, done, trunc, info = env.step(action)
    x = env.render()
    print(x)

    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
env.close()



