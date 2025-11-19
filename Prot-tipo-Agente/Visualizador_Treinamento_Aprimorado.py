"""
Visualizador de Treinamento Melhorado - Sistema Adaptativo de Labirintos
Incorpora modelo de animação antigo e mostra melhores rodadas
"""

import random
import math
import heapq
import time
import csv
from collections import deque, defaultdict, namedtuple
from typing import Optional, List, Dict, Tuple, Set

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib não disponível. Instale com: pip install matplotlib")
    exit(1)

# Importa o sistema melhorado (corrigido)
from Labirinto_adaptativo_improved import (
    Maze, astar, QLearningAgent, MazeEnv, DifficultyController, evaluate, run_episode
)


# ==================== FUNÇÃO DE ANIMAÇÃO HERDADA DO ARQUIVO ANTIGO ====================
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
    # Cores: branco (livre), verde claro (início), azul (caminho), amarelo (goal), vermelho (agente), preto (parede)
    cores = ['white', 'lightgreen', 'lightblue', 'yellow', 'red', 'black']
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


# ==================== VISUALIZADOR APRIMORADO ====================
class AdvancedTrainingVisualizer:
    """Visualizador avançado com seleção das melhores rodadas"""
    
    def __init__(self, maze_size: int = 21, branching: float = 0.1):
        self.maze_size = maze_size
        self.branching = branching
        self.q_agent = QLearningAgent(alpha=0.5, gamma=0.95, eps=0.4, eps_decay=0.995, eps_min=0.05)
        
        # Histórico de métricas
        self.episodes_data = []
        self.episodes = []
        self.q_steps = []
        self.astar_steps = []
        self.success_rates = []
        self.epsilons = []
        
        # Melhores episódios
        self.best_astar_episodes = []
        self.best_q_episodes = []
        self.best_episode_count = 3  # Número de melhores episódios a manter
        
    def train_episode(self, seed: Optional[int] = None) -> dict:
        """Treina um episódio e retorna métricas completas"""
        # Gera labirinto
        maze = Maze(self.maze_size, self.maze_size, 
                   branching=self.branching, seed=seed)
        grid = maze.generate()
        start, goal = maze.start, maze.goal
        
        # Caminho ótimo com A*
        astar_path = astar(grid, start, goal)
        if not astar_path:
            return None
        
        astar_steps = len(astar_path) - 1
        
        # Treina Q-Learning
        env = MazeEnv(grid, start, goal)
        state_pos = env.reset()
        state = self.q_agent.get_state(state_pos, goal)
        q_path = [state_pos]
        
        for _ in range(env.max_steps):
            valid = env.valid_actions()
            if not valid:
                break
            
            action = self.q_agent.choose(state, valid, training=True)
            new_pos, reward, done = env.step(action)
            new_state = self.q_agent.get_state(new_pos, goal)
            
            self.q_agent.learn(state, action, reward, new_state, 
                             env.valid_actions())
            
            state = new_state
            q_path.append(new_pos)
            
            if done:
                break
        
        success = env.pos == goal
        if success:
            self.q_agent.decay_epsilon()
        
        q_steps = env.steps
        
        # Calcula eficiência
        efficiency = q_steps / astar_steps if astar_steps > 0 else float('inf')
        
        episode_data = {
            'episode': len(self.episodes_data) + 1,
            'seed': seed,
            'grid': grid,
            'start': start,
            'goal': goal,
            'astar_path': astar_path,
            'astar_steps': astar_steps,
            'q_path': q_path,
            'q_steps': q_steps,
            'success': success,
            'efficiency': efficiency,
            'epsilon': self.q_agent.eps
        }
        
        # Adiciona às melhores performances
        self._update_best_episodes(episode_data)
        
        return episode_data
    
    def _update_best_episodes(self, episode_data: dict):
        """Atualiza as listas de melhores episódios"""
        # Atualiza melhores episódios A* (sempre ótimos)
        if len(self.best_astar_episodes) < self.best_episode_count:
            self.best_astar_episodes.append(episode_data)
        else:
            # Substitui o pior se o atual for melhor
            worst_efficiency = min(ep['efficiency'] for ep in self.best_astar_episodes)
            if episode_data['efficiency'] < worst_efficiency:
                worst_idx = self.best_astar_episodes.index(
                    next(ep for ep in self.best_astar_episodes if ep['efficiency'] == worst_efficiency)
                )
                self.best_astar_episodes[worst_idx] = episode_data
        
        # Atualiza melhores episódios Q-Learning (apenas sucessos)
        if episode_data['success']:
            if len(self.best_q_episodes) < self.best_episode_count:
                self.best_q_episodes.append(episode_data)
            else:
                # Substitui o pior se o atual for melhor
                worst_efficiency = max(ep['efficiency'] for ep in self.best_q_episodes)
                if episode_data['efficiency'] < worst_efficiency:
                    worst_idx = self.best_q_episodes.index(
                        next(ep for ep in self.best_q_episodes if ep['efficiency'] == worst_efficiency)
                    )
                    self.best_q_episodes[worst_idx] = episode_data
    
    def visualize_training_progress(self, num_episodes: int = 30):
        """Visualiza o progresso do treinamento em tempo real"""
        print(f"\n🎯 Iniciando treinamento por {num_episodes} episódios...")
        print(f"📊 Tamanho do labirinto: {self.maze_size}x{self.maze_size}")
        print(f"🔀 Branching: {self.branching}")
        
        # Configuração dos gráficos
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sistema Adaptativo: Progresso do Treinamento', 
                    fontsize=16, fontweight='bold')
        
        # Cores
        color_astar = '#2ecc71'  # Verde
        color_q = '#e74c3c'      # Vermelho
        color_success = '#3498db'  # Azul
        
        # Subplot 1: Comparação de passos
        ax1 = axes[0, 0]
        ax1.set_title('Número de Passos por Episódio', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Passos')
        line_astar, = ax1.plot([], [], color=color_astar, linewidth=3, 
                              marker='o', markersize=6, label='A* (Ótimo)', alpha=0.8)
        line_q, = ax1.plot([], [], color=color_q, linewidth=3,
                          marker='s', markersize=6, label='Q-Learning', alpha=0.8)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Taxa de sucesso
        ax2 = axes[0, 1]
        ax2.set_title('Taxa de Sucesso (Janela Móvel)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episódio')
        ax2.set_ylabel('Taxa de Sucesso')
        ax2.set_ylim([0, 1.1])
        line_success, = ax2.plot([], [], color=color_success, linewidth=3, marker='o', markersize=6)
        ax2.axhline(y=1.0, color=color_astar, linestyle='--', linewidth=2,
                   label='A* (100%)', alpha=0.7)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Epsilon (exploração)
        ax3 = axes[1, 0]
        ax3.set_title('Epsilon (Nível de Exploração)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episódio')
        ax3.set_ylabel('Epsilon')
        ax3.set_ylim([0, max(0.5, self.q_agent.eps)])
        line_eps, = ax3.plot([], [], color='#f39c12', linewidth=3, marker='^', markersize=6)
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Eficiência relativa
        ax4 = axes[1, 1]
        ax4.set_title('Eficiência: Q-Learning vs A*', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Episódio')
        ax4.set_ylabel('Razão (Q/A*)')
        line_ratio, = ax4.plot([], [], color='#9b59b6', linewidth=3, marker='D', markersize=6)
        ax4.axhline(y=1.0, color=color_astar, linestyle='--', linewidth=2,
                   label='Ótimo', alpha=0.7)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Função de atualização dos gráficos
        window_size = 10
        
        def update_plots():
            if not self.episodes:
                return
                
            episodes = list(range(1, len(self.episodes) + 1))
            
            # Subplot 1: Passos
            line_astar.set_data(episodes, self.astar_steps)
            line_q.set_data(episodes, self.q_steps)
            ax1.relim()
            ax1.autoscale_view()
            
            # Subplot 2: Taxa de sucesso (janela móvel)
            if len(self.success_rates) >= window_size:
                moving_avg = [
                    sum(self.success_rates[max(0, i-window_size):i]) / 
                    min(window_size, i)
                    for i in range(1, len(self.success_rates) + 1)
                ]
                line_success.set_data(episodes, moving_avg)
                ax2.relim()
                ax2.autoscale_view()
            
            # Subplot 3: Epsilon
            line_eps.set_data(episodes, self.epsilons)
            ax3.relim()
            ax3.autoscale_view()
            
            # Subplot 4: Razão
            ratios = [q / a if a > 0 else 0 
                     for q, a in zip(self.q_steps, self.astar_steps)]
            line_ratio.set_data(episodes, ratios)
            ax4.relim()
            ax4.autoscale_view()
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        # Inicia o modo interativo
        plt.ion()
        plt.show()
        
        # Loop de treinamento
        for ep in range(num_episodes):
            result = self.train_episode(seed=random.randint(0, 1 << 30))
            
            if result is None:
                print(f"❌ Episódio {ep+1}: Labirinto sem solução, pulando...")
                continue
            
            # Armazena métricas
            self.episodes.append(ep + 1)
            self.astar_steps.append(result['astar_steps'])
            self.q_steps.append(result['q_steps'])
            self.success_rates.append(1.0 if result['success'] else 0.0)
            self.epsilons.append(result['epsilon'])
            self.episodes_data.append(result)
            
            # Print progresso
            status = "✅" if result['success'] else "❌"
            print(f"📝 {status} Ep {ep+1:2d}/{num_episodes} | "
                  f"A*={result['astar_steps']:3d} | "
                  f"Q={result['q_steps']:3d} | "
                  f"Eficiência={result['efficiency']:.2f}x | "
                  f"ε={result['epsilon']:.3f}")
            
            # Atualiza gráficos
            if (ep + 1) % 5 == 0 or ep == num_episodes - 1:
                update_plots()
        
        # Finaliza
        print(f"\n🎉 Treinamento concluído!")
        self._print_final_stats()
        
        plt.ioff()
        plt.show()
        
        return self.best_astar_episodes, self.best_q_episodes
    
    def _print_final_stats(self):
        """Imprime estatísticas finais"""
        if not self.episodes_data:
            return
            
        final_window = min(10, len(self.success_rates))
        final_success_rate = sum(self.success_rates[-final_window:]) / final_window
        
        final_ratios = [
            q / a for q, a in 
            zip(self.q_steps[-final_window:], self.astar_steps[-final_window:])
        ]
        avg_final_ratio = sum(final_ratios) / len(final_ratios)
        
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"   • Taxa de sucesso (últimos {final_window}): {final_success_rate:.1%}")
        print(f"   • Eficiência média: {avg_final_ratio:.2f}x o caminho ótimo")
        print(f"   • Epsilon final: {self.q_agent.eps:.3f}")
        print(f"   • Episódios treinados: {len(self.episodes_data)}")
    
    def show_best_episodes(self, delay: int = 150):
        """Mostra as melhores rodadas de cada agente"""
        print(f"\n🏆 Mostrando as melhores performances...")
        
        # Mostra melhores episódios A*
        if self.best_astar_episodes:
            print(f"\n🌟 MELHORES EPISÓDIOS A* ({len(self.best_astar_episodes)}):")
            for i, ep in enumerate(self.best_astar_episodes, 1):
                title = f"A* - Episódio {ep['episode']} | {ep['astar_steps']} passos | Eficiência: {ep['efficiency']:.2f}x"
                print(f"   {i}. {title}")
                animar_episodio(
                    ep['grid'], ep['start'], ep['goal'], 
                    ep['astar_path'], title, delay
                )
        
        # Mostra melhores episódios Q-Learning
        if self.best_q_episodes:
            print(f"\n🤖 MELHORES EPISÓDIOS Q-LEARNING ({len(self.best_q_episodes)}):")
            for i, ep in enumerate(self.best_q_episodes, 1):
                title = f"Q-Learning - Episódio {ep['episode']} | {ep['q_steps']} passos | Eficiência: {ep['efficiency']:.2f}x"
                print(f"   {i}. {title}")
                animar_episodio(
                    ep['grid'], ep['start'], ep['goal'], 
                    ep['q_path'], title, delay
                )
        else:
            print("\n⚠️  Nenhum episódio Q-Learning bem-sucedido para mostrar.")
    
    def compare_episodes(self, index1: int, index2: int):
        """Compara dois episódios lado a lado"""
        if (index1 < 1 or index1 > len(self.episodes_data) or 
            index2 < 1 or index2 > len(self.episodes_data)):
            print("❌ Índices de episódio inválidos!")
            return
        
        ep1 = self.episodes_data[index1 - 1]
        ep2 = self.episodes_data[index2 - 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Comparação: Episódio {index1} vs Episódio {index2}', fontsize=16)
        
        # Episódio 1
        ax1.set_title(f"Episódio {index1}\nA*: {ep1['astar_steps']} passos | Q: {ep1['q_steps']} passos", 
                     fontsize=12)
        
        # Criar visualização do episódio 1
        grid1 = ep1['grid']
        path1 = set(ep1['q_path'])
        text1 = self._create_maze_text(grid1, ep1['start'], ep1['goal'], path1)
        ax1.text(0.05, 0.95, text1, fontfamily='monospace', fontsize=6,
                verticalalignment='top', transform=ax1.transAxes)
        ax1.axis('off')
        
        # Episódio 2
        ax2.set_title(f"Episódio {index2}\nA*: {ep2['astar_steps']} passos | Q: {ep2['q_steps']} passos", 
                     fontsize=12)
        
        # Criar visualização do episódio 2
        grid2 = ep2['grid']
        path2 = set(ep2['q_path'])
        text2 = self._create_maze_text(grid2, ep2['start'], ep2['goal'], path2)
        ax2.text(0.05, 0.95, text2, fontfamily='monospace', fontsize=6,
                verticalalignment='top', transform=ax2.transAxes)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _create_maze_text(self, grid: List[List[int]], start: Tuple[int, int], 
                         goal: Tuple[int, int], path: Set[Tuple[int, int]]) -> str:
        """Cria representação textual do labirinto"""
        lines = []
        for y in range(len(grid)):
            row = []
            for x in range(len(grid[0])):
                if (y, x) == start:
                    row.append("S")
                elif (y, x) == goal:
                    row.append("G")
                elif (y, x) in path:
                    row.append("·")
                elif grid[y][x] == 1:
                    row.append("█")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)


def run_adaptive_training_demo():
    """Demonstração completa do sistema adaptativo"""
    print("🚀 VISUALIZADOR AVANÇADO - SISTEMA ADAPTATIVO DE LABIRINTOS")
    print("=" * 70)
    
    # Configurações
    print("\n📋 Configurações:")
    maze_sizes = [15, 17, 21]
    branchings = [0.08, 0.12, 0.15]
    
    print("1. Labirinto pequeno (15x15, branching=0.08)")
    print("2. Labirinto médio (17x17, branching=0.12)")
    print("3. Labirinto grande (21x21, branching=0.15)")
    
    choice = input("\nEscolha o nível (1-3): ").strip()
    
    if choice == "1":
        maze_size, branching = 15, 0.08
    elif choice == "2":
        maze_size, branching = 17, 0.12
    elif choice == "3":
        maze_size, branching = 21, 0.15
    else:
        print("❌ Opção inválida. Usando configurações padrão.")
        maze_size, branching = 17, 0.12
    
    # Cria visualizador
    visualizer = AdvancedTrainingVisualizer(maze_size=maze_size, branching=branching)
    
    # Pergunta sobre número de episódios
    num_episodes = input(f"\nNúmero de episódios para treinar (padrão=25): ").strip()
    try:
        num_episodes = int(num_episodes) if num_episodes else 25
    except ValueError:
        num_episodes = 25
    
    # Treinamento
    print(f"\n🎯 Iniciando treinamento...")
    best_astar, best_q = visualizer.visualize_training_progress(num_episodes)
    
    # Mostra melhores performances
    show_best = input("\n🎬 Deseja ver as melhores performances animadas? (s/N): ").strip().lower()
    if show_best in ['s', 'sim', 'y', 'yes']:
        delay = input("Velocidade da animação em ms (padrão=150): ").strip()
        try:
            delay = int(delay) if delay else 150
        except ValueError:
            delay = 150
        visualizer.show_best_episodes(delay)
    
    # Comparação opcional
    compare = input("\n🔍 Deseja comparar dois episódios específicos? (s/N): ").strip().lower()
    if compare in ['s', 'sim', 'y', 'yes']:
        try:
            ep1 = int(input("Índice do primeiro episódio: "))
            ep2 = int(input("Índice do segundo episódio: "))
            visualizer.compare_episodes(ep1, ep2)
        except ValueError:
            print("❌ Índices inválidos.")
    
    print("\n🎉 Demonstração concluída!")


if __name__ == "__main__":
    if not HAS_MATPLOTLIB:
        print("Este script requer matplotlib!")
        print("Instale com: pip install matplotlib")
    else:
        run_adaptive_training_demo()