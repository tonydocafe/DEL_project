import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Drone3DEnv(gym.Env):
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        
        # definition of observation and action space
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Conectar ao PyBullet
        self.physics_client = p.connect(p.DIRECT)  # Usando p.DIRECT
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.drone = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # Configuração de gravidade

        # Criação do drone com URDF e ajustes
        self.drone = p.loadURDF("cube.urdf", basePosition=[0, 0, 1], globalScaling=0.5)
        p.changeDynamics(self.drone, -1, mass=1.0)  # Ajuste na massa do drone (1.0 kg)

        # Estado inicial (posição + velocidade linear)
        return np.array([0, 0, 1, 0, 0, 0], dtype=np.float32), {}

    def step(self, action):
        # Aumentando a intensidade das forças aplicadas ao drone
        forces = action * 50  # Ajustando a intensidade da força (pode ser mais alto)

        # Aplica as forças em todos os eixos (x, y, z)
        p.applyExternalForce(self.drone, -1, forceObj=[forces[0], forces[1], forces[2]], 
                            posObj=[0, 0, 0], flags=p.LINK_FRAME)

        # Passa a simulação para frente
        p.stepSimulation()

        # Obtenção da posição e velocidade do drone
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        lin_vel, _ = p.getBaseVelocity(self.drone)

        # Observações
        obs = np.array([*pos, *lin_vel], dtype=np.float32)

        # Recompensa com base na posição Z (altura)
        reward = -abs(pos[2] - 2)  # Recompensa negativa se não estiver em altura desejada
        terminated = abs(pos[2] - 2) < 0.1  # Termina se estiver perto de altura 2

        return obs, reward, terminated, False, {}

    def render(self):
        pass  # A renderização no gráfico 3D será feita fora do método render

    def close(self):
        p.disconnect()

# ==========================
# Teste com visualização do movimento 3D
# ==========================
env = Drone3DEnv(render_mode="rgb_array")
obs, _ = env.reset()
positions = []

# Definir alvo (em algum ponto no espaço)
target = np.array([5, 5, 2])  # Exemplo de alvo fixo em (5, 5, 2)

# Simulação para capturar os movimentos do drone
for _ in range(100):
    action = env.action_space.sample()  # Ações aleatórias para movimentar o drone
    obs, reward, terminated, _, _ = env.step(action)

    # Captura da posição do drone
    pos = obs[:3]
    positions.append(pos)

    if terminated:
        break

env.close()

# Plotar o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extrair as posições do drone (X, Y, Z)
x_pos = [pos[0] for pos in positions]
y_pos = [pos[1] for pos in positions]
z_pos = [pos[2] for pos in positions]

# Plotar o movimento do drone
ax.plot(x_pos, y_pos, z_pos, label="Movimento do Drone", color='blue')

# Plotar o alvo
ax.scatter(target[0], target[1], target[2], color='red', label="Alvo")

# Adicionar rótulos e título
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Movimento do Drone no Espaço 3D')

# Exibir o gráfico
plt.legend()
plt.show()
