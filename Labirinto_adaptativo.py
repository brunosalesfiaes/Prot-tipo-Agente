

import random
import math
import heapq
import time
import csv
from collections import deque, defaultdict, namedtuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback para mediana sem numpy
    def median(lst):
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n == 0:
            return 0.0
        if n % 2 == 0:
            return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2.0
        return float(sorted_lst[n//2])


# ---------- Gerador de Labirinto ----------
class Maze:
    def __init__(self, width, height, branching=0.0, obstacle_density=0.0, seed=None):
        assert width % 2 == 1 and height % 2 == 1, "Use tamanhos ímpares para células coerentes."
        self.width = width
        self.height = height
        self.branching = float(branching)
        self.obstacle_density = float(obstacle_density)
        self.rng = random.Random(seed)
        self.grid = self._make_empty()
        self.start = (1, 1)
        self.goal = (height-2, width-2)

    def _make_empty(self):
        return [[1 for _ in range(self.width)] for _ in range(self.height)]

    def generate(self):
        # Backtracking recursivo em células ímpares
        self.grid = self._make_empty()
        H, W = self.height, self.width
        stack = []
        start = (1, 1)
        self.grid[start[0]][start[1]] = 0
        stack.append(start)
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        while stack:
            cell = stack[-1]
            neighbors = []
            for dy, dx in dirs:
                ny, nx = cell[0]+dy, cell[1]+dx
                if 0 < ny < H and 0 < nx < W and self.grid[ny][nx] == 1:
                    neighbors.append((ny, nx))
            if neighbors:
                nxt = self.rng.choice(neighbors)
                # derruba parede entre as células
                wy, wx = (cell[0]+nxt[0])//2, (cell[1]+nxt[1])//2
                self.grid[wy][wx] = 0
                self.grid[nxt[0]][nxt[1]] = 0
                stack.append(nxt)
            else:
                stack.pop()

        # Adiciona loops / ramificações removendo paredes aleatoriamente com probabilidade branching
        # itera sobre paredes internas e remove aleatoriamente
        for y in range(1, H-1):
            for x in range(1, W-1):
                if self.grid[y][x] == 1:
                    # parede entre dois corredores? Verifica se remover conecta dois corredores previamente desconectados (cria loop)
                    if (y % 2 == 1 and x % 2 == 0) or (y % 2 == 0 and x % 2 == 1):
                        if self.rng.random() < self.branching:
                            # remove parede
                            self.grid[y][x] = 0

        # Adiciona obstáculos aleatórios para aumentar dificuldade se necessário
        free_cells = [(y, x) for y in range(1, H-1) for x in range(1, W-1) 
                     if self.grid[y][x] == 0 and (y, x) != self.start and (y, x) != self.goal]
        n_obs = int(len(free_cells) * self.obstacle_density)
        if n_obs > 0 and free_cells:
            for (y, x) in self.rng.sample(free_cells, min(n_obs, len(free_cells))):
                self.grid[y][x] = 1

        # Garante que início e objetivo estão livres
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0
        return self.grid

    def as_text(self, path=None):
        # path: conjunto de (y,x)
        out = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (y, x) == self.start:
                    row.append("S")
                elif (y, x) == self.goal:
                    row.append("G")
                elif path and (y, x) in path:
                    row.append(".")
                elif self.grid[y][x] == 1:
                    row.append("#")
                else:
                    row.append(" ")
            out.append("".join(row))
        return "\n".join(out)


# ---------- Agente A* (determinístico) ----------
def astar(grid, start, goal):
    H, W = len(grid), len(grid[0])
    
    def neighbors(node):
        y, x = node
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 0:
                yield (ny, nx)
    
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    
    while open_heap:
        f, g, node, parent = heapq.heappop(open_heap)
        if node in came_from:
            continue
        came_from[node] = parent
        if node == goal:
            # reconstrói caminho
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
            heapq.heappush(open_heap, (tentative_g + heuristic(nb, goal), tentative_g, nb, node))
    return None


# ---------- Agente Q-learning (tabela) ----------
class QLearningAgent:
    def __init__(self, actions=[(0, 1), (1, 0), (0, -1), (-1, 0)], alpha=0.5, gamma=0.9, eps=0.2):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q = defaultdict(float)  # Q[(estado, indice_acao)] = valor

    def state_from_pos(self, pos):
        # representação do estado é simplesmente a posição (y,x)
        return pos

    def choose(self, state, valid_actions):
        # epsilon-greedy entre ações válidas
        if random.random() < self.eps:
            return random.choice(valid_actions) if valid_actions else (0, 1)
        # ganancioso
        best = None
        best_val = -1e9
        for ai in valid_actions:
            v = self.q[(state, ai)]
            if v > best_val:
                best_val = v
                best = ai
        if best is None:
            return random.choice(valid_actions) if valid_actions else (0, 1)
        return best

    def learn(self, s, a, r, s2, valid_actions_s2):
        key = (s, a)
        # max_a' Q(s2,a') - máximo Q do próximo estado
        max_q_s2 = max([self.q[(s2, a2)] for a2 in valid_actions_s2]) if valid_actions_s2 else 0.0
        self.q[key] += self.alpha * (r + self.gamma * max_q_s2 - self.q[key])


# ---------- Wrapper do Ambiente ----------
class MazeEnv:
    def __init__(self, grid, start, goal, max_steps=None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.pos = start
        self.max_steps = max_steps if max_steps is not None else len(grid)*len(grid[0])*4
        self.steps = 0

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.pos

    def valid_actions(self):
        H, W = len(self.grid), len(self.grid[0])
        valid = []
        for i, (dy, dx) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
            ny, nx = self.pos[0]+dy, self.pos[1]+dx
            if 0 <= ny < H and 0 <= nx < W and self.grid[ny][nx] == 0:
                valid.append((dy, dx))
        return valid

    def step(self, action):
        dy, dx = action
        ny, nx = self.pos[0]+dy, self.pos[1]+dx
        if not (0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0])) or self.grid[ny][nx] == 1:
            # colidiu com parede: penaliza levemente e permanece
            self.steps += 1
            done = False
            return self.pos, -0.5, done  # pequena penalidade
        self.pos = (ny, nx)
        self.steps += 1
        if self.pos == self.goal:
            return self.pos, 10.0, True
        if self.steps >= self.max_steps:
            return self.pos, -5.0, True
        return self.pos, -0.01, False  # pequeno custo por passo para incentivar caminhos mais curtos


# ---------- Controlador Adaptativo de Dificuldade ----------
class DifficultyController:
    def __init__(self, width=21, height=21,
                 branching=0.02, obstacle_density=0.0,
                 branching_limits=(0.0, 0.5), size_limits=(11, 101)):
        self.width = width
        self.height = height
        self.branching = branching
        self.obstacle_density = obstacle_density
        self.branching_min, self.branching_max = branching_limits
        self.size_min, self.size_max = size_limits
        self.history = []

    def current_params(self):
        return {
            'width': self.width,
            'height': self.height,
            'branching': self.branching,
            'obstacle_density': self.obstacle_density
        }

    def update(self, recent_stats):
        """
        recent_stats: dicionário com chaves:
            - median_steps: passos medianos
            - success_rate: taxa de sucesso
            - median_ratio: razão (passos / caminho_mínimo)
        Usa heurísticas simples para ajustar branching e tamanho.
        """
        median_steps = recent_stats['median_steps']
        success = recent_stats['success_rate']
        median_ratio = recent_stats.get('median_ratio', 1.0)

        # comportamento alvo: queremos median_ratio em torno de target_ratio
        target_ratio = 1.5
        # se agente está resolvendo muito rápido (median_ratio < target_low) -> aumenta branching (mais difícil)
        if median_ratio < 1.2 and success > 0.8:
            self.branching = min(self.branching * 1.2 + 0.01, self.branching_max)
            # ocasionalmente aumenta tamanho
            if random.random() < 0.2 and self.width+2 <= self.size_max:
                self.width = min(self.width + 2, self.size_max)
                self.height = min(self.height + 2, self.size_max)
        # se agente está com dificuldade -> reduz branching e tamanho
        if median_ratio > 3.0 or success < 0.5:
            self.branching = max(self.branching * 0.7 - 0.01, self.branching_min)
            if random.random() < 0.3 and self.width-2 >= self.size_min:
                self.width = max(self.width - 2, self.size_min)
                self.height = max(self.height - 2, self.size_min)

        # pequeno ruído aleatório ocasional para explorar espaço de parâmetros
        if random.random() < 0.05:
            self.branching = min(self.branching_max, max(self.branching_min, 
                                                         self.branching + (random.random()-0.5)*0.05))

        self.branching = max(self.branching_min, min(self.branching, self.branching_max))
        self.history.append(self.current_params())
        return self.current_params()


# ---------- Executor / Treinador ----------
def run_episode(generator_params, agent_type='astar', q_agent=None, render=False, seed=None):
    maze = Maze(generator_params['width'], generator_params['height'],
                branching=generator_params['branching'],
                obstacle_density=generator_params.get('obstacle_density', 0.0),
                seed=seed)
    grid = maze.generate()
    start, goal = maze.start, maze.goal

    if agent_type == 'astar':
        path = astar(grid, start, goal)
        if path is None:
            return {'success': False, 'steps': None, 'shortest': None, 'path': None, 'grid': grid, 'start': start, 'goal': goal}
        steps = len(path)-1
        return {'success': True, 'steps': steps, 'shortest': steps, 'path': path, 'grid': grid, 'start': start, 'goal': goal}

    elif agent_type == 'qlearning':
        env = MazeEnv(grid, start, goal)
        state = env.reset()
        total_steps = 0
        max_steps = env.max_steps
        path = [state]  # Rastreia o caminho percorrido
        # Executa um único episódio usando a política do q_agent (com aprendizado)
        for t in range(max_steps):
            valid = env.valid_actions()
            if not valid:
                break
            action = q_agent.choose(state, valid)
            new_state, reward, done = env.step(action)
            # Dá ao agente a chance de aprender durante a execução
            q_agent.learn(state, action, reward, new_state, env.valid_actions())
            state = new_state
            path.append(state)  # Adiciona nova posição ao caminho
            total_steps += 1
            if done:
                return {'success': env.pos == goal, 'steps': total_steps, 'shortest': None, 'path': path, 'grid': grid, 'start': start, 'goal': goal}
        return {'success': False, 'steps': total_steps, 'shortest': None, 'path': path, 'grid': grid, 'start': start, 'goal': goal}
    else:
        raise ValueError("agent_type inválido")


def evaluate(controller, episodes=50, agent_mode='astar', q_agent=None, log_csv=None):
    results = []
    for ep in range(episodes):
        params = controller.current_params()
        seed = random.randint(0, 1 << 30)
        r = run_episode(params, agent_type=agent_mode, q_agent=q_agent, seed=seed)
        # Se A*, obtém caminho mais curto
        if agent_mode == 'astar' and r['success']:
            shortest = r['shortest']
            steps = r['steps']
            ratio = steps / shortest if shortest else float('inf')
        else:
            shortest = r['shortest']
            steps = r['steps'] if r['steps'] is not None else 9999
            ratio = None

        results.append({'ep': ep, 'success': bool(r['success']), 'steps': steps, 
                       'shortest': shortest, 'ratio': ratio, 'params': params.copy(),
                       'grid': r.get('grid'), 'path': r.get('path'), 
                       'start': r.get('start'), 'goal': r.get('goal')})
    
    # calcula estatísticas
    successes = sum(1 for x in results if x['success'])
    success_rate = successes / len(results)
    steps_list = [x['steps'] for x in results if x['steps'] is not None and x['steps'] < 9999]
    
    if HAS_NUMPY:
        median_steps = float(np.median(steps_list)) if steps_list else float('inf')
    else:
        median_steps = float(median(steps_list)) if steps_list else float('inf')
    
    ratios = [x['ratio'] for x in results if x['ratio'] is not None]
    
    if HAS_NUMPY:
        median_ratio = float(np.median(ratios)) if ratios else None
    else:
        median_ratio = float(median(ratios)) if ratios else None

    stats = {'success_rate': success_rate, 'median_steps': median_steps, 
             'median_ratio': median_ratio, 'results': results}
    return stats


# ---------- Animação com Matplotlib ----------
def animar_episodio(grid, start, goal, path, title="Simulação de Agente em Labirinto", delay=100):
    """
    Anima o agente percorrendo o caminho no labirinto.
    path: lista de tuplas (y, x) representando o caminho percorrido
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib não está disponível. Instale com: pip install matplotlib")
        return
    
    if not path:
        print("Nenhum caminho para animar.")
        return
    
    H, W = len(grid), len(grid[0])
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.axis("off")
    
    # Converte path de set para lista se necessário, e garante ordem
    if isinstance(path, set):
        # Se for set, tenta reconstruir ordem usando BFS do start ao goal
        # Isso garante um caminho válido mesmo que não seja o original
        from collections import deque
        path_set = path
        q = deque([start])
        parent = {start: None}
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        found = False
        while q:
            node = q.popleft()
            if node == goal:
                found = True
                break
            for dy, dx in dirs:
                ny, nx = node[0] + dy, node[1] + dx
                if (ny, nx) in path_set and (ny, nx) not in parent:
                    # Valida que não é parede
                    if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 0:
                        parent[(ny, nx)] = node
                        q.append((ny, nx))
        if found:
            # Reconstrói caminho
            path_list = []
            cur = goal
            while cur is not None:
                path_list.append(cur)
                cur = parent[cur]
            path_list.reverse()
        else:
            # Fallback: apenas lista os pontos válidos (não paredes)
            path_list = [start]
            for p in path_set:
                if p != start and p != goal:
                    py, px = p
                    if 0 <= py < H and 0 <= px < W and grid[py][px] == 0:
                        path_list.append(p)
            path_list.append(goal)
    else:
        # Filtra o caminho para remover quaisquer posições que sejam paredes
        path_list = []
        for p in path:
            py, px = p
            if 0 <= py < H and 0 <= px < W:
                # Sempre inclui start e goal, e células livres
                if p == start or p == goal or grid[py][px] == 0:
                    path_list.append(p)
                # Se for parede, pula (não adiciona ao caminho)
    
    # Cria colormap customizado para melhor visualização
    from matplotlib.colors import ListedColormap
    # Cores: branco (livre), verde claro (início), azul (caminho), amarelo (goal), vermelho (agente), preto (parede)
    cores = ['white', 'lightgreen', 'lightblue', 'yellow', 'red', 'black']
    n_bins = 6
    cmap_custom = ListedColormap(cores)
    
    # Cria o mapa de cores base (0=livre, 1=parede)
    # Valores: 0=livre (branco), 1=início (verde), 2=caminho (azul), 3=goal (amarelo), 4=agente (vermelho), 5=parede (preto)
    imagem = [[5 if grid[r][c] == 1 else 0 for c in range(W)] for r in range(H)]
    img_plot = ax.imshow(imagem, cmap=cmap_custom, vmin=0, vmax=5)
    
    # Função de atualização da animação
    def update(frame):
        if frame >= len(path_list):
            return [img_plot]
        
        r, c = path_list[frame]
        # Cria cópia do grid base - garante que só marca caminho em células livres (não paredes)
        temp = [[5 if grid[i][j] == 1 else 0 for j in range(W)] for i in range(H)]
        
        # Marca o caminho percorrido até agora (apenas em células livres)
        for idx in range(frame):
            if idx < len(path_list):
                pr, pc = path_list[idx]
                # Só marca se não for parede e não for início/goal
                if grid[pr][pc] == 0 and (pr, pc) != start and (pr, pc) != goal:
                    temp[pr][pc] = 2  # caminho percorrido (azul)
        
        # Marca início e fim (sempre visíveis, apenas se não forem paredes)
        sr, sc = start
        gr, gc = goal
        if grid[sr][sc] == 0:  # Verifica se é livre
            temp[sr][sc] = 1   # início (verde claro)
        if grid[gr][gc] == 0:  # Verifica se é livre
            temp[gr][gc] = 3   # goal (amarelo)
        
        # Marca posição atual do agente (apenas se não for parede)
        if grid[r][c] == 0:  # Verifica se é livre
            temp[r][c] = 4  # agente (vermelho)
        
        img_plot.set_data(temp)
        return [img_plot]
    
    ani = animation.FuncAnimation(fig, update, frames=len(path_list),
                                  interval=delay, repeat=False, blit=True)
    plt.show()


# ---------- Demo / Principal ----------
def main_demo():
    random.seed(1234)
    controller = DifficultyController(width=21, height=21, branching=0.02, obstacle_density=0.0)
    qagent = QLearningAgent(alpha=0.3, gamma=0.95, eps=0.2)

    # Alterna: usa A* para medição (oráculo), mas também deixa Q-agent treinar entre episódios.
    n_rounds = 30
    episodios_para_animar = []  # Armazena episódios interessantes para animar
    
    for round_idx in range(n_rounds):
        # Avalia um lote com A* para medir distribuição do caminho mais curto
        stats = evaluate(controller, episodes=20, agent_mode='astar')
        print(f"[Rodada {round_idx}] ANTES da atualizacao: taxa_sucesso={stats['success_rate']:.2f}, "
              f"passos_medianos={stats['median_steps']:.2f}, razao_mediana={stats['median_ratio']}")
        
        # Guarda um episódio A* interessante para animar (primeiro sucesso da rodada)
        for result in stats['results']:
            if result['success'] and result.get('path') is not None:
                episodios_para_animar.append({
                    'round': round_idx,
                    'type': 'astar',
                    'grid': result.get('grid'),
                    'start': result.get('start', (1, 1)),
                    'goal': result.get('goal'),
                    'path': result['path'],
                    'title': f"Rodada {round_idx} - A* (Passos: {result['steps']})"
                })
                break
        
        # Atualiza controlador de dificuldade com estatísticas recentes
        controller.update({'median_steps': stats['median_steps'], 
                         'success_rate': stats['success_rate'], 
                         'median_ratio': stats['median_ratio'] or 1.0})
        params = controller.current_params()
        print(f" -> Novos parametros: largura={params['width']} altura={params['height']} "
              f"branching={params['branching']:.3f}")

        # Treina agente Q-learning em alguns episódios na dificuldade atual
        for t in range(30):
            run_episode(params, agent_type='qlearning', q_agent=qagent, 
                       seed=random.randint(0, 1 << 30))

        # Avalia desempenho do agente Q-learning
        q_stats = evaluate(controller, episodes=15, agent_mode='qlearning', q_agent=qagent)
        # q_stats não terá razões de caminho mais curto, mas fornece taxa_sucesso e passos_medianos
        print(f" Agente Q-Learning apos treinamento: taxa_sucesso={q_stats['success_rate']:.2f}, "
              f"passos_medianos={q_stats['median_steps']:.1f}")
        
        # Guarda um episódio Q-learning interessante para animar (primeiro sucesso da rodada)
        for result in q_stats['results']:
            if result['success'] and result.get('path') is not None:
                # Obtém start e goal do resultado ou usa padrão do Maze
                start_pos = result.get('start', (1, 1))
                goal_pos = result.get('goal')
                if goal_pos is None:
                    H, W = len(result.get('grid', [])), len(result.get('grid', [[]])[0]) if result.get('grid') else 0
                    if H > 0 and W > 0:
                        goal_pos = (H-2, W-2)
                if goal_pos is not None:
                    episodios_para_animar.append({
                        'round': round_idx,
                        'type': 'qlearning',
                        'grid': result.get('grid'),
                        'start': start_pos,
                        'goal': goal_pos,
                        'path': result['path'],
                        'title': f"Rodada {round_idx} - Q-Learning (Passos: {result['steps']})"
                    })
                    break

    print("\n=== Demo finalizado. Iniciando animações ===")
    
    # Anima os episódios coletados
    if HAS_MATPLOTLIB and episodios_para_animar:
        print(f"Animando {len(episodios_para_animar)} episódios...")
        for i, ep in enumerate(episodios_para_animar):
            # Se não temos start/goal, tenta obter do grid
            if ep['start'] is None or ep['goal'] is None:
                # Assume padrão: start em (1,1) e goal no canto oposto
                H, W = len(ep['grid']), len(ep['grid'][0])
                ep['start'] = (1, 1)
                ep['goal'] = (H-2, W-2)
            
            print(f"Animando episódio {i+1}/{len(episodios_para_animar)}: {ep['title']}")
            animar_episodio(ep['grid'], ep['start'], ep['goal'], ep['path'], 
                          title=ep['title'], delay=150)
    else:
        if not HAS_MATPLOTLIB:
            print("Matplotlib não está disponível. Instale com: pip install matplotlib")
        else:
            print("Nenhum episódio para animar.")


if __name__ == "__main__":
    main_demo()

