import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

class DroneEnv3D(gym.Env):
    def __init__(self):
        super(DroneEnv3D, self).__init__()
        
        # definition of observation and action space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -10, -5, -5, -5], dtype=np.float32),
            high=np.array([10, 10, 10, 5, 5, 5], dtype=np.float32),
            dtype=np.float32
        )
        
        self.dt = 0.1  # environment update time 
        self.max_steps = 100
        self.current_step = 0
        
        # initial state where v is the velocity [x, y, z, vx, vy, vz]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.target = np.array([5.0, 5.0, 5.0], dtype=np.float32)  # Posição alvo
        
        # create graphic
        self.fig, self.ax = plt.subplots()
        
    def _get_obs(self):
        return self.state
        
    # Euclidean distance, because the movement is free
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self.state[:3] - self.target),
            "velocity": np.linalg.norm(self.state[3:])
        }
    
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        
        
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        
        acceleration = action * 0.5  # Limitar aceleração
        
        # update speed 
        self.state[3:] += acceleration * self.dt
        self.state[3:] = np.clip(self.state[3:], -5, 5)  # limit speed 
        
        # update position
        self.state[:3] += self.state[3:] * self.dt
        
        # check position
        self.state[:3] = np.clip(self.state[:3], -10, 10)
        
        # Calcular recompensa
        distance = np.linalg.norm(self.state[:3] - self.target)
        reward = -distance  # Recompensa negativa proporcional à distância
        
        # Verificar término do episódio
        self.current_step += 1
        terminated = distance < 0.5 or self.current_step >= self.max_steps
        truncated = False
        
        # Info adicional
        info = self._get_info()
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        self.ax.clear()
        
        # Determinar tamanho do drone baseado no Z (quanto maior Z, menor o ponto)
        z = self.state[2]
        size = max(20, 100 - (z * 5))  # O tamanho diminui conforme o Z aumenta
        
        # Plotar drone
        self.ax.scatter(self.state[0], self.state[1], c='red', s=size, label='Drone')
        
        # Plotar alvo
        self.ax.scatter(self.target[0], self.target[1], c='green', s=100, marker='x', label='Alvo')
        
        # Configurar gráfico
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_title(f"Posição do Drone (Z={z:.2f})")

        display.clear_output(wait=True)
        display.display(self.fig)
        plt.close()


# Criar instância do ambiente
env = DroneEnv3D()

# Executar um episódio com ações aleatórias
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Ação aleatória
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()
