"""
Exemplo rápido de uso do sistema adaptativo de labirintos.
Demonstração simplificada para entender o funcionamento básico.
"""

from adaptive_maze import Maze, astar, QLearningAgent, MazeEnv, DifficultyController, run_episode, evaluate

def exemplo_basico():
    """Exemplo básico: gerar um labirinto e resolver com A*."""
    print("="*60)
    print("EXEMPLO 1: Geração e Resolução de Labirinto")
    print("="*60)
    
    # Criar labirinto
    maze = Maze(width=15, height=15, branching=0.1, obstacle_density=0.0, seed=42)
    grid = maze.generate()
    start, goal = maze.start, maze.goal
    
    print(f"\nLabirinto gerado: {maze.width}x{maze.height}")
    print(f"Início: {start}, Objetivo: {goal}")
    
    # Resolver com A*
    path = astar(grid, start, goal)
    if path:
        print(f"\n[OK] Caminho encontrado! Comprimento: {len(path)-1} passos")
        print("\nVisualização do labirinto (primeiras 10 linhas):")
        print(maze.as_text(path=set(path))[:200])
    else:
        print("\n[ERRO] Nenhum caminho encontrado!")
    
    return maze, path


def exemplo_qlearning():
    """Exemplo: treinar agente Q-Learning em um labirinto."""
    print("\n" + "="*60)
    print("EXEMPLO 2: Treinamento de Agente Q-Learning")
    print("="*60)
    
    # Criar labirinto menor para treinamento rápido
    maze = Maze(width=11, height=11, branching=0.05, seed=123)
    grid = maze.generate()
    start, goal = maze.start, maze.goal
    
    # Criar agente e ambiente
    agent = QLearningAgent(alpha=0.3, gamma=0.95, eps=0.3)
    env = MazeEnv(grid, start, goal)
    
    # Treinar por alguns episódios
    print(f"\nTreinando agente em labirinto {maze.width}x{maze.height}...")
    sucessos = 0
    for ep in range(50):
        state = env.reset()
        total_reward = 0
        for step in range(200):
            valid = env.valid_actions()
            if not valid:
                break
            action = agent.choose(state, valid)
            new_state, reward, done = env.step(action)
            agent.learn(state, action, reward, new_state, env.valid_actions())
            state = new_state
            total_reward += reward
            if done:
                if env.pos == goal:
                    sucessos += 1
                break
    
    print(f"Taxa de sucesso após 50 episódios: {sucessos/50:.2%}")
    return agent


def exemplo_controlador():
    """Exemplo: uso do controlador adaptativo."""
    print("\n" + "="*60)
    print("EXEMPLO 3: Controlador Adaptativo (3 rodadas)")
    print("="*60)
    
    controller = DifficultyController(width=15, height=15, branching=0.02)
    agent = QLearningAgent(alpha=0.3, gamma=0.95, eps=0.2)
    
    for round_idx in range(3):
        print(f"\n--- Rodada {round_idx+1} ---")
        params = controller.current_params()
        print(f"Parâmetros atuais: largura={params['width']}, altura={params['height']}, "
              f"branching={params['branching']:.3f}")
        
        # Avaliar com A*
        stats = evaluate(controller, episodes=10, agent_mode='astar')
        print(f"A* - Sucesso: {stats['success_rate']:.2%}, "
              f"Passos médios: {stats['median_steps']:.1f}")
        
        # Atualizar controlador
        controller.update({
            'median_steps': stats['median_steps'],
            'success_rate': stats['success_rate'],
            'median_ratio': stats['median_ratio'] or 1.0
        })
        
        # Treinar Q-agent
        for _ in range(20):
            run_episode(params, agent_type='qlearning', q_agent=agent, 
                       seed=None)
        
        # Avaliar Q-agent
        q_stats = evaluate(controller, episodes=10, agent_mode='qlearning', 
                          q_agent=agent)
        print(f"Q-Learning - Sucesso: {q_stats['success_rate']:.2%}, "
              f"Passos médios: {q_stats['median_steps']:.1f}")
        
        # Mostrar novos parâmetros
        new_params = controller.current_params()
        if new_params != params:
            print(f">> Parametros ajustados: branching={new_params['branching']:.3f}")


def main():
    """Executa todos os exemplos."""
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO DO SISTEMA ADAPTATIVO DE LABIRINTOS")
    print("="*60)
    
    try:
        # Exemplo 1: Básico
        exemplo_basico()
        
        # Exemplo 2: Q-Learning
        exemplo_qlearning()
        
        # Exemplo 3: Controlador
        exemplo_controlador()
        
        print("\n" + "="*60)
        print("[OK] Todos os exemplos executados com sucesso!")
        print("="*60)
        print("\nPara executar o sistema completo, rode: python adaptive_maze.py")
        
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

