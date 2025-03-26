import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import display  

class Drone3DEnv(gym.Env):
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        
        # definition of observation and actions space
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # connect to PyBullet
        self.physics_client = p.connect(p.DIRECT)  # use p.DIRECT to run without window
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.drone = None

        
        self.target = np.array([5, 5, 4])  

        
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.positions = []  # path taken

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  
        # drone in PyBullet
        self.drone = p.loadURDF("cube.urdf", basePosition=[0, 0, 1], globalScaling=0.5)
        p.changeDynamics(self.drone, -1, mass=1.0)  

        self.positions = [] 
        return np.array([0, 0, 1, 0, 0, 0], dtype=np.float32), {}

    def step(self, action):
        forces = action * 50  # applied force
        p.applyExternalForce(self.drone, -1, forceObj=[forces[0], forces[1], forces[2]], 
                            posObj=[0, 0, 0], flags=p.LINK_FRAME)

        p.stepSimulation()

        # obtaining the drone position and speed
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        lin_vel, _ = p.getBaseVelocity(self.drone)

        # update position
        self.positions.append(pos)

        obs = np.array([*pos, *lin_vel], dtype=np.float32)
        reward = -np.linalg.norm(pos - self.target)  # penalizes target distance
        terminated = np.linalg.norm(pos - self.target) < 0.2  # end if close to target

        return obs, reward, terminated, False, {}

    def render(self):
      self.ax.clear()  
      
     
      if self.positions:
          x_pos = [pos[0] for pos in self.positions]
          y_pos = [pos[1] for pos in self.positions]
          z_pos = [pos[2] for pos in self.positions]

          self.ax.plot(x_pos, y_pos, z_pos, label="Trajetória do Drone", color='blue')

      
      self.ax.scatter(self.target[0], self.target[1], self.target[2], 
                      color='red', marker='x', s=100, label="Alvo")

      
      self.ax.set_xlim(-10, 10)
      self.ax.set_ylim(-10, 10)
      self.ax.set_zlim(0, 5)
      self.ax.set_xlabel('X')
      self.ax.set_ylabel('Y')
      self.ax.set_zlabel('Z')
      self.ax.set_title('Movimento do Drone 3D')
      self.ax.legend()

      display.clear_output(wait=True)  #update
      display.display(self.fig) #view
      plt.close(self.fig)  



env = Drone3DEnv()
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample() 
    obs, reward, terminated, _, _ = env.step(action)

    env.render()  

    if terminated:
        break

env.close()
