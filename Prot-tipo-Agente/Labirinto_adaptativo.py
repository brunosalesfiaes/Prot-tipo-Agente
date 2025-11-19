"""
Sistema Adaptativo de Labirintos - Versão Melhorada
Implementa geração procedural com ajuste dinâmico de dificuldade
usando A* como oráculo e Q-Learning aprimorado como aprendiz.
"""

import random
import math
import heapq
import csv
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    def median(lst):
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 0:
            return (sorted_lst[n//2-1] + sorted_lst[n//2]) / 2.0
        return float(sorted_lst[n//2])


# ==================== GERADOR DE LABIRINTO ====================
class Maze:
    """Gerador procedural de labirintos usando Recursive Backtracker"""
    
    def __init__(self, width: int, height: int, branching: float = 0.0, 
                 obstacle_density: float = 0.0, seed: Optional[int] = None):
        assert width % 2 == 1 and height % 2 == 1, "Use tamanhos ímpares"
        self.width = width
        self.height = height
        self.branching = max(0.0, min(1.0, float(branching)))
        self.obstacle_density = max(0.0, min(0.3, float(obstacle_density)))
        self.rng = random.Random(seed)
        self.grid = []
        self.start = (1, 1)
        self.goal = (height-2, width-2)
    
    def generate(self) -> List[List[int]]:
        """Gera o labirinto (0=livre, 1=parede)"""
        # Inicializa com paredes
        self.grid = [[1] * self.width for _ in range(self.height)]
        
        # Recursive Backtracker
        stack = [self.start]
        self.grid[self.start[0]][self.start[1]] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        
        while stack:
            cell = stack[-1]
            # Busca vizinhos não visitados
            neighbors = []
            for dy, dx in dirs:
                ny, nx = cell[0] + dy, cell[1] + dx
                if (0 <= ny < self.height and 0 <= nx < self.width 
                    and self.grid[ny][nx] == 1):
                    neighbors.append((ny, nx))
            
            if neighbors:
                nxt = self.rng.choice(neighbors)
                # Remove parede entre células
                wy, wx = (cell[0] + nxt[0]) // 2, (cell[1] + nxt[1]) // 2
                self.grid[wy][wx] = 0
                self.grid[nxt[0]][nxt[1]] = 0
                stack.append(nxt)
            else:
                stack.pop()
        
        # Adiciona loops (aumenta branching)
        self._add_loops()
        
        # Adiciona obstáculos aleatórios
        self._add_obstacles()
        
        # Garante que início e objetivo estão livres
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0
        
        return self.grid
    
    def _add_loops(self):
        """Adiciona loops removendo paredes internas"""
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y][x] == 1:
                    # Parede horizontal ou vertical
                    is_horizontal = (y % 2 == 0 and x % 2 == 1)
                    is_vertical = (y % 2 == 1 and x % 2 == 0)
                    
                    if (is_horizontal or is_vertical) and self.rng.random() < self.branching:
                        # Verifica se conecta dois corredores
                        if self._connects_corridors(y, x):
                            self.grid[y][x] = 0
    
    def _connects_corridors(self, y: int, x: int) -> bool:
        """Verifica se remover parede conecta corredores"""
        if y % 2 == 0:  # Parede horizontal
            return (self.grid[y-1][x] == 0 and self.grid[y+1][x] == 0)
        else:  # Parede vertical
            return (self.grid[y][x-1] == 0 and self.grid[y][x+1] == 0)
    
    def _add_obstacles(self):
        """Adiciona obstáculos aleatórios em células livres"""
        free_cells = [
            (y, x) for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if (self.grid[y][x] == 0 and 
                (y, x) != self.start and 
                (y, x) != self.goal)
        ]
        
        n_obs = int(len(free_cells) * self.obstacle_density)
        if n_obs > 0 and free_cells:
            for y, x in self.rng.sample(free_cells, min(n_obs, len(free_cells))):
                self.grid[y][x] = 1
    
    def as_text(self, path: Optional[Set[Tuple[int, int]]] = None) -> str:
        """Retorna representação textual do labirinto"""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (y, x) == self.start:
                    row.append("S")
                elif (y, x) == self.goal:
                    row.append("G")
                elif path and (y, x) in path:
                    row.append("·")
                elif self.grid[y][x] == 1:
                    row.append("█")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)


# ==================== AGENTE A* (ORÁCULO) ====================
def astar(grid: List[List[int]], start: Tuple[int, int], 
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Busca A* - retorna caminho ótimo ou None"""
    H, W = len(grid), len(grid[0])
    
    def neighbors(node):
        y, x = node
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 0:
                yield (ny, nx)
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_heap = [(heuristic(start, goal), 0, start, None)]
    came_from = {}
    gscore = {start: 0}
    
    while open_heap:
        f, g, node, parent = heapq.heappop(open_heap)
        
        if node in came_from:
            continue
        
        came_from[node] = parent
        
        if node == goal:
            # Reconstrói caminho
            path = []
            cur = node
            while cur:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path
        
        for nb in neighbors(node):
            tentative_g = g + 1
            if nb in gscore and tentative_g >= gscore[nb]:
                continue
            
            gscore[nb] = tentative_g
            f_score = tentative_g + heuristic(nb, goal)
            heapq.heappush(open_heap, (f_score, tentative_g, nb, node))
    
    return None


# ==================== AGENTE Q-LEARNING MELHORADO ====================
class QLearningAgent:
    """
    Q-Learning com melhorias:
    - Estado enriquecido (posição + distância ao objetivo)
    - Decaimento de epsilon
    - Replay de experiências prioritárias
    """
    
    def __init__(self, alpha: float = 0.5, gamma: float = 0.9, 
                 eps: float = 0.3, eps_decay: float = 0.995, 
                 eps_min: float = 0.05):
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # direita, baixo, esquerda, cima
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.q = defaultdict(float)
        self.visit_count = defaultdict(int)  # Contagem de visitas
        self.episodes_trained = 0
    
    def get_state(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> Tuple:
        """Estado enriquecido: posição + distância Manhattan ao objetivo"""
        dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        # Discretiza distância em bins
        dist_bin = min(dist // 5, 10)  # 0-4, 5-9, 10-14, ..., 50+
        return (pos, dist_bin)
    
    def choose(self, state: Tuple, valid_actions: List[Tuple[int, int]], 
               training: bool = True) -> Tuple[int, int]:
        """Epsilon-greedy com bonus de exploração (UCB)"""
        if not valid_actions:
            return (0, 1)
        
        eps_current = self.eps if training else 0.0  # Sem exploração na avaliação
        
        if random.random() < eps_current:
            return random.choice(valid_actions)
        
        # Greedy com UCB (Upper Confidence Bound)
        best_action = None
        best_value = float('-inf')
        
        total_visits = sum(self.visit_count[(state, a)] for a in valid_actions) + 1
        
        for action in valid_actions:
            key = (state, action)
            q_val = self.q[key]
            
            # Bonus de exploração UCB
            visits = self.visit_count[key] + 1
            exploration_bonus = 0.5 * math.sqrt(math.log(total_visits) / visits)
            
            value = q_val + exploration_bonus
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action if best_action else random.choice(valid_actions)
    
    def learn(self, state: Tuple, action: Tuple[int, int], reward: float,
              next_state: Tuple, valid_next_actions: List[Tuple[int, int]]):
        """Atualização Q-Learning"""
        key = (state, action)
        self.visit_count[key] += 1
        
        # Max Q(s', a')
        max_q_next = max(
            [self.q[(next_state, a)] for a in valid_next_actions],
            default=0.0
        )
        
        # Atualização temporal difference
        td_error = reward + self.gamma * max_q_next - self.q[key]
        self.q[key] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decai epsilon após cada episódio"""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self.episodes_trained += 1


# ==================== AMBIENTE ====================
class MazeEnv:
    """Ambiente de labirinto para RL"""
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int],
                 goal: Tuple[int, int], max_steps: Optional[int] = None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.pos = start
        self.max_steps = max_steps or (len(grid) * len(grid[0]) * 2)
        self.steps = 0
    
    def reset(self) -> Tuple[int, int]:
        """Reinicia ambiente"""
        self.pos = self.start
        self.steps = 0
        return self.pos
    
    def valid_actions(self) -> List[Tuple[int, int]]:
        """Retorna ações válidas na posição atual"""
        H, W = len(self.grid), len(self.grid[0])
        valid = []
        
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = self.pos[0] + dy, self.pos[1] + dx
            if 0 <= ny < H and 0 <= nx < W and self.grid[ny][nx] == 0:
                valid.append((dy, dx))
        
        return valid
    
    def step(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """Executa ação e retorna (novo_estado, recompensa, done)"""
        dy, dx = action
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        
        # Verifica colisão com parede
        if (not (0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0])) 
            or self.grid[ny][nx] == 1):
            self.steps += 1
            return self.pos, -1.0, False  # Penalidade por colidir
        
        # Move agente
        old_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        self.pos = (ny, nx)
        new_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        self.steps += 1
        
        # Recompensas
        if self.pos == self.goal:
            return self.pos, 100.0, True  # Grande recompensa por alcançar objetivo
        
        if self.steps >= self.max_steps:
            return self.pos, -10.0, True  # Penalidade por timeout
        
        # Recompensa por aproximar do objetivo
        reward = -0.1 + (old_dist - new_dist) * 0.3
        
        return self.pos, reward, False


# ==================== CONTROLADOR ADAPTATIVO ====================
class DifficultyController:
    """
    Controlador adaptativo de dificuldade usando PID-like control
    """
    
    def __init__(self, width: int = 21, height: int = 21,
                 branching: float = 0.02, obstacle_density: float = 0.0,
                 target_ratio: float = 1.4):
        self.width = width
        self.height = height
        self.branching = branching
        self.obstacle_density = obstacle_density
        self.target_ratio = target_ratio  # Razão alvo (passos_agente / caminho_ótimo)
        
        # Limites
        self.branching_limits = (0.0, 0.5)
        self.size_limits = (11, 101)
        
        # Histórico
        self.history = []
        self.error_integral = 0.0
        self.last_error = 0.0
        
        # Ganhos PID
        self.kp = 0.15  # Proporcional
        self.ki = 0.02  # Integral
        self.kd = 0.05  # Derivativo
    
    def current_params(self) -> Dict:
        """Retorna parâmetros atuais"""
        return {
            'width': self.width,
            'height': self.height,
            'branching': self.branching,
            'obstacle_density': self.obstacle_density
        }
    
    def update(self, stats: Dict):
        """Atualiza parâmetros baseado em estatísticas"""
        success_rate = stats['success_rate']
        median_ratio = stats.get('median_ratio', 1.0)
        
        # Controle PID para branching
        error = self.target_ratio - median_ratio
        self.error_integral += error
        error_derivative = error - self.last_error
        
        # Sinal de controle PID
        control = (self.kp * error + 
                  self.ki * self.error_integral + 
                  self.kd * error_derivative)
        
        # Atualiza branching
        self.branching += control * 0.05
        self.branching = max(self.branching_limits[0], 
                           min(self.branching, self.branching_limits[1]))
        
        # Ajuste de tamanho baseado em taxa de sucesso
        if success_rate > 0.9 and median_ratio < 1.3:
            # Muito fácil - aumenta tamanho
            if self.width + 4 <= self.size_limits[1]:
                self.width += 2
                self.height += 2
        elif success_rate < 0.4:
            # Muito difícil - reduz tamanho
            if self.width - 4 >= self.size_limits[0]:
                self.width -= 2
                self.height -= 2
        
        self.last_error = error
        self.history.append({
            **self.current_params(),
            'success_rate': success_rate,
            'median_ratio': median_ratio,
            'error': error
        })
        
        return self.current_params()


# ==================== EXECUTOR ====================
def run_episode(params: Dict, agent_type: str = 'astar',
                q_agent: Optional[QLearningAgent] = None,
                seed: Optional[int] = None,
                training: bool = True) -> Dict:
    """Executa um episódio único"""
    maze = Maze(params['width'], params['height'],
                branching=params['branching'],
                obstacle_density=params.get('obstacle_density', 0.0),
                seed=seed)
    grid = maze.generate()
    start, goal = maze.start, maze.goal
    
    if agent_type == 'astar':
        path = astar(grid, start, goal)
        if path is None:
            return {
                'success': False, 'steps': None, 'shortest': None,
                'path': None, 'grid': grid, 'start': start, 'goal': goal
            }
        return {
            'success': True, 'steps': len(path) - 1, 'shortest': len(path) - 1,
            'path': path, 'grid': grid, 'start': start, 'goal': goal
        }
    
    elif agent_type == 'qlearning':
        env = MazeEnv(grid, start, goal)
        state_pos = env.reset()
        state = q_agent.get_state(state_pos, goal)
        path = [state_pos]
        total_reward = 0.0
        
        for _ in range(env.max_steps):
            valid = env.valid_actions()
            if not valid:
                break
            
            action = q_agent.choose(state, valid, training=training)
            new_pos, reward, done = env.step(action)
            new_state = q_agent.get_state(new_pos, goal)
            
            if training:
                q_agent.learn(state, action, reward, new_state, env.valid_actions())
            
            state = new_state
            path.append(new_pos)
            total_reward += reward
            
            if done:
                break
        
        success = env.pos == goal
        if training and success:
            q_agent.decay_epsilon()
        
        return {
            'success': success, 'steps': env.steps, 'shortest': None,
            'path': path, 'grid': grid, 'start': start, 'goal': goal,
            'reward': total_reward
        }
    
    raise ValueError(f"agent_type inválido: {agent_type}")


def evaluate(controller: DifficultyController, episodes: int = 50,
             agent_mode: str = 'astar',
             q_agent: Optional[QLearningAgent] = None) -> Dict:
    """Avalia agente por múltiplos episódios"""
    results = []
    
    for ep in range(episodes):
        params = controller.current_params()
        seed = random.randint(0, 1 << 30)
        
        # Avaliação sem treino
        r = run_episode(params, agent_type=agent_mode, q_agent=q_agent,
                       seed=seed, training=False)
        
        # Calcula métricas
        if agent_mode == 'astar' and r['success']:
            shortest = r['shortest']
            steps = r['steps']
            ratio = steps / shortest if shortest else float('inf')
        else:
            # Para Q-Learning, obtém caminho ótimo com A*
            if r['success']:
                optimal_path = astar(r['grid'], r['start'], r['goal'])
                shortest = len(optimal_path) - 1 if optimal_path else None
                steps = r['steps']
                ratio = steps / shortest if shortest and shortest > 0 else None
            else:
                shortest = None
                steps = r['steps'] if r['steps'] is not None else 9999
                ratio = None
        
        results.append({
            'ep': ep, 'success': r['success'], 'steps': steps,
            'shortest': shortest, 'ratio': ratio
        })
    
    # Estatísticas
    successes = sum(1 for x in results if x['success'])
    success_rate = successes / len(results)
    
    steps_list = [x['steps'] for x in results 
                 if x['steps'] is not None and x['steps'] < 9999]
    if HAS_NUMPY:
        median_steps = float(np.median(steps_list)) if steps_list else float('inf')
    else:
        median_steps = float(median(steps_list)) if steps_list else float('inf')
    
    ratios = [x['ratio'] for x in results if x['ratio'] is not None]
    if HAS_NUMPY:
        median_ratio = float(np.median(ratios)) if ratios else None
    else:
        median_ratio = float(median(ratios)) if ratios else None
    
    return {
        'success_rate': success_rate,
        'median_steps': median_steps,
        'median_ratio': median_ratio,
        'results': results
    }


# ==================== MAIN ====================
def main():
    """Execução principal do sistema adaptativo"""
    print("="*70)
    print("SISTEMA ADAPTATIVO DE LABIRINTOS - Versão Melhorada")
    print("="*70)
    
    random.seed(42)
    
    # Inicialização
    controller = DifficultyController(width=15, height=15, branching=0.02,
                                     target_ratio=1.5)
    q_agent = QLearningAgent(alpha=0.4, gamma=0.95, eps=0.4,
                            eps_decay=0.995, eps_min=0.05)
    
    n_rounds = 25
    csv_file = 'adaptive_results_improved.csv'
    
    # CSV header
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'width', 'height', 'branching',
                        'astar_success', 'astar_median_steps', 'astar_median_ratio',
                        'q_success', 'q_median_steps', 'q_median_ratio',
                        'q_epsilon', 'q_episodes_trained'])
    
    print(f"\nExecutando {n_rounds} rodadas de treinamento adaptativo...\n")
    
    for round_idx in range(n_rounds):
        params = controller.current_params()
        
        print(f"[Rodada {round_idx + 1}/{n_rounds}]")
        print(f"  Parâmetros: {params['width']}x{params['height']}, "
              f"branching={params['branching']:.3f}")
        
        # 1. Mede dificuldade com A* (oráculo)
        astar_stats = evaluate(controller, episodes=20, agent_mode='astar')
        print(f"  A* (Oráculo): sucesso={astar_stats['success_rate']:.1%}, "
              f"passos={astar_stats['median_steps']:.1f}, "
              f"razão={astar_stats['median_ratio']:.2f}")
        
        # 2. Treina Q-Learning
        for _ in range(40):
            run_episode(params, agent_type='qlearning', q_agent=q_agent,
                       seed=random.randint(0, 1 << 30), training=True)
        
        # 3. Avalia Q-Learning
        q_stats = evaluate(controller, episodes=20, agent_mode='qlearning',
                          q_agent=q_agent)
        
        # CORREÇÃO: Verificar se os valores são None antes de formatar
        median_ratio_str = f"{q_stats['median_ratio']:.2f}" if q_stats['median_ratio'] is not None else "N/A"
        
        print(f"  Q-Learning: sucesso={q_stats['success_rate']:.1%}, "
              f"passos={q_stats['median_steps']:.1f}, "
              f"razão={median_ratio_str}, "
              f"ε={q_agent.eps:.3f}")
        
        # 4. Atualiza controlador (usa A* como referência)
        controller.update({
            'median_steps': astar_stats['median_steps'],
            'success_rate': astar_stats['success_rate'],
            'median_ratio': astar_stats['median_ratio'] or 1.0
        })
        
        # 5. Log CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_idx, params['width'], params['height'], params['branching'],
                astar_stats['success_rate'], astar_stats['median_steps'],
                astar_stats['median_ratio'] or 0,
                q_stats['success_rate'], q_stats['median_steps'],
                q_stats['median_ratio'] or 0,
                q_agent.eps, q_agent.episodes_trained
            ])
        
        print()
    
    print("="*70)
    print(f"[✓] Treinamento concluído!")
    print(f"    Resultados salvos em: {csv_file}")
    print(f"    Q-Learning treinou {q_agent.episodes_trained} episódios")
    print(f"    ε final: {q_agent.eps:.3f}")
    print("="*70)
    print("\nExecute 'python analisar_resultados.py' para visualizar os resultados")


if __name__ == "__main__":
    main()