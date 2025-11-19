#!/usr/bin/env python3
"""
Demonstração Rápida - Sistema Adaptativo de Labirintos
Executa uma demonstração simples para verificar se tudo funciona
"""

import random
import sys

def demonstracao_rapida():
    """Executa uma demonstração rápida do sistema"""
    print("="*60)
    print("🚀 DEMONSTRAÇÃO RÁPIDA - SISTEMA ADAPTATIVO")
    print("="*60)
    
    try:
        # Importa o sistema
        print("📦 Importando sistema...")
        import Labirinto_adaptativo_improved as sistema
        print("✅ Sistema importado com sucesso!")
        
        # Cria um labirinto
        print("\n🏗️  Gerando labirinto...")
        maze = sistema.Maze(15, 15, branching=0.1, seed=42)
        grid = maze.generate()
        print(f"✅ Labirinto {maze.width}x{maze.height} gerado!")
        print(f"   Início: {maze.start} | Objetivo: {maze.goal}")
        
        # Testa A*
        print("\n🧠 Executando A* (oráculo)...")
        caminho_astar = sistema.astar(grid, maze.start, maze.goal)
        if caminho_astar:
            print(f"✅ A* encontrou caminho com {len(caminho_astar)-1} passos")
        else:
            print("❌ A* não encontrou caminho")
            return False
        
        # Testa Q-Learning rapidamente
        print("\n🤖 Treinando Q-Learning (10 episódios)...")
        agente = sistema.QLearningAgent(alpha=0.5, gamma=0.9, eps=0.3)
        sucessos = 0
        
        for ep in range(10):
            env = sistema.MazeEnv(grid, maze.start, maze.goal)
            estado = env.reset()
            estado_q = agente.get_state(estado, maze.goal)
            
            for _ in range(100):  # Máximo 100 passos
                acoes_validas = env.valid_actions()
                if not acoes_validas:
                    break
                
                acao = agente.choose(estado_q, acoes_validas, training=True)
                novo_estado, recompensa, done = env.step(acao)
                novo_estado_q = agente.get_state(novo_estado, maze.goal)
                
                agente.learn(estado_q, acao, recompensa, novo_estado_q, env.valid_actions())
                estado_q = novo_estado_q
                
                if done:
                    break
            
            if env.pos == maze.goal:
                sucessos += 1
                agente.decay_epsilon()
        
        print(f"✅ Q-Learning treinou 10 episódios")
        print(f"   Sucessos: {sucessos}/10 ({sucessos/10:.0%})")
        print(f"   Epsilon final: {agente.eps:.3f}")
        
        # Testa avaliação
        print("\n📊 Avaliando agente treinado...")
        controlador = sistema.DifficultyController(15, 15, 0.1)
        estatisticas = sistema.evaluate(
            controlador,
            episodes=5, agent_mode='qlearning', q_agent=agente
        )
        
        print(f"✅ Avaliação concluída:")
        print(f"   Taxa de sucesso: {estatisticas['success_rate']:.1%}")
        print(f"   Passos médios: {estatisticas['median_steps']:.1f}")
        
        # Mostra representação textual do labirinto
        print("\n🗺️  Visualização do labirinto:")
        print(caminho_astar)
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("✅ Todos os componentes estão funcionando corretamente")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NA DEMONSTRAÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("Iniciando demonstração rápida do sistema...")
    
    sucesso = demonstracao_rapida()
    
    if sucesso:
        print("\n💡 Para executar o treinamento completo:")
        print("   python Labirinto_adaptativo.py")
        print("\n💡 Para visualizar resultados:")
        print("   python analisar_resultados.py")
        print("\n💡 Para executar testes:")
        print("   python teste_correcoes.py")
    else:
        print("\n⚠️  Houve problemas na demonstração.")
        print("   Verifique se todas as correções foram aplicadas.")
    
    return 0 if sucesso else 1

if __name__ == "__main__":
    sys.exit(main())