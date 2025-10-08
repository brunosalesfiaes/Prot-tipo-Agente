import random
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def gerar_labirinto(linhas, colunas, densidade_parede=0.25, max_attempts=1000):
    """Gera um labirinto (0 = livre, 1 = parede) garantindo que start e goal estejam conectados"""
    attempts = 0
    while attempts < max_attempts:
        grid = [[1 if random.random() < densidade_parede else 0 for _ in range(colunas)] for _ in range(linhas)]
        start = (0, 0)
        goal = (linhas - 1, colunas - 1)
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0
        if existe_caminho_bfs(grid, start, goal):
            return grid, start, goal
        attempts += 1
    raise RuntimeError("Não foi possível gerar um labirinto conectado em várias tentativas.")

def existe_caminho_bfs(grid, start, goal):
    """Verifica se existe caminho com BFS"""
    linhas, colunas = len(grid), len(grid[0])
    q = deque([start])
    visitado = set([start])
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        r,c = q.popleft()
        if (r,c) == goal:
            return True
        for dr,dc in dirs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < linhas and 0 <= nc < colunas and grid[nr][nc] == 0 and (nr,nc) not in visitado:
                visitado.add((nr,nc))
                q.append((nr,nc))
    return False

def bfs_caminho(grid, start, goal):
    """Retorna lista com caminho do start até goal (inclusive) usando BFS"""
    linhas, colunas = len(grid), len(grid[0])
    q = deque([start])
    pai = {start: None}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        r,c = q.popleft()
        if (r,c) == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = pai[cur]
            path.reverse()
            return path
        for dr,dc in dirs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < linhas and 0 <= nc < colunas and grid[nr][nc] == 0 and (nr,nc) not in pai:
                pai[(nr,nc)] = (r,c)
                q.append((nr,nc))
    return None

def simular_agente_visual(grid, start, goal, delay=100):
    """Anima o agente percorrendo o caminho encontrado com BFS"""
    caminho = bfs_caminho(grid, start, goal)
    if not caminho:
        print("Nenhum caminho encontrado.")
        return
    
    linhas, colunas = len(grid), len(grid[0])
    fig, ax = plt.subplots()
    ax.set_title("Simulação de Agente em Labirinto")
    ax.axis("off")

    # Cria o mapa de cores base
    imagem = [[1 if grid[r][c] == 1 else 0 for c in range(colunas)] for r in range(linhas)]
    img_plot = ax.imshow(imagem, cmap="binary")

    # Função de atualização da animação
    def update(frame):
        r, c = caminho[frame]
        temp = [[1 if grid[i][j] == 1 else 0 for j in range(colunas)] for i in range(linhas)]
        
        # Marca o caminho percorrido
        for (pr, pc) in caminho[:frame]:
            temp[pr][pc] = 0.5  # azul claro (caminho)
        
        # Marca início e fim
        sr, sc = start
        gr, gc = goal
        temp[sr][sc] = 0.3   # verde (início)
        temp[gr][gc] = 0.8   # dourado (goal)

        # Marca posição atual do agente
        temp[r][c] = 0.9  # vermelho

        img_plot.set_data(temp)
        return [img_plot]

    ani = animation.FuncAnimation(fig, update, frames=len(caminho),
                                  interval=delay, repeat=False)
    plt.show()

# Execução do programa
if __name__ == "__main__":
    linhas = 15
    colunas = 25
    densidade_parede = 0.30  # 0.0 = livre, 0.5 = muito bloqueado
    grid, start, goal = gerar_labirinto(linhas, colunas, densidade_parede)
    simular_agente_visual(grid, start, goal, delay=150)
