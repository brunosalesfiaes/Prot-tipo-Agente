"""
Visualizador de Resultados - Sistema Adaptativo de Labirintos
Mostra as melhores rodadas baseadas em resultados de treinamento
"""

import csv
import random
import os
from typing import List, Dict, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Importa o sistema
from Labirinto_adaptativo_improved import (
    Maze, astar, QLearningAgent, MazeEnv
)

# Importa a função de animação
from Visualizador_Treinamento_Aprimorado import animar_episodio


class ResultsVisualizer:
    """Visualizador de resultados baseado em dados de treinamento"""
    
    def __init__(self, csv_file: str = 'adaptive_results_improved.csv'):
        self.csv_file = csv_file
        self.results_data = []
        self.best_episodes = {
            'astar': [],
            'qlearning': []
        }
        self.load_training_results()
    
    def load_training_results(self):
        """Carrega dados do arquivo CSV de treinamento"""
        if not os.path.exists(self.csv_file):
            print(f"❌ Arquivo {self.csv_file} não encontrado!")
            print("Execute primeiro: python Labirinto_adaptativo.py")
            return
        
        print(f"📊 Carregando resultados de {self.csv_file}...")
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Converte valores numéricos
                for key in row:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
                self.results_data.append(row)
        
        print(f"✅ Carregados {len(self.results_data)} resultados")
    
    def analyze_performance(self):
        """Analisa o desempenho geral"""
        if not self.results_data:
            print("❌ Nenhum dado para analisar")
            return
        
        print("\n📈 ANÁLISE DE PERFORMANCE GERAL")
        print("=" * 50)
        
        # Estatísticas gerais
        total_rounds = len(self.results_data)
        avg_q_success = sum(r['q_success'] for r in self.results_data) / total_rounds
        best_q_success = max(r['q_success'] for r in self.results_data)
        
        final_round = self.results_data[-1]
        
        print(f"🏁 Total de rodadas: {total_rounds}")
        print(f"📊 Taxa de sucesso Q-Learning média: {avg_q_success:.1%}")
        print(f"🎯 Melhor taxa de sucesso Q-Learning: {best_q_success:.1%}")
        print(f"🏆 Taxa de sucesso final: {final_round['q_success']:.1%}")
        
        # Evolução do epsilon
        epsilons = [r['q_epsilon'] for r in self.results_data]
        print(f"🔍 Epsilon inicial: {epsilons[0]:.3f}")
        print(f"🔍 Epsilon final: {epsilons[-1]:.3f}")
        
        # Evolução dos parâmetros
        first = self.results_data[0]
        print(f"\n⚙️  EVOLUÇÃO DOS PARÂMETROS:")
        print(f"   Tamanho inicial: {first['width']:.0f}x{first['height']:.0f}")
        print(f"   Tamanho final: {final_round['width']:.0f}x{final_round['height']:.0f}")
        print(f"   Branching inicial: {first['branching']:.3f}")
        print(f"   Branching final: {final_round['branching']:.3f}")
    
    def generate_sample_episodes(self, num_best: int = 5):
        """Gera episódios de exemplo para visualização das melhores performances"""
        if not self.results_data:
            return
        
        print(f"\n🎬 Gerando {num_best} melhores episódios para visualização...")
        
        # Pega os episódios com melhores taxas de sucesso
        sorted_results = sorted(self.results_data, key=lambda x: x['q_success'], reverse=True)
        
        for i, result in enumerate(sorted_results[:num_best]):
            round_num = int(result['round'])
            success_rate = result['q_success']
            
            print(f"🔄 Gerando episódio {i+1} (Rodada {round_num}, Sucesso: {success_rate:.1%})")
            
            # Recria o labirinto com os parâmetros da rodada
            maze = Maze(
                int(result['width']), 
                int(result['height']), 
                branching=result['branching'],
                seed=42 + round_num  # Seed baseada na rodada para reproducibility
            )
            grid = maze.generate()
            
            # Executa episódios para A* e Q-Learning
            astar_result = self._run_single_episode(maze, 'astar')
            qlearning_result = self._run_single_episode(maze, 'qlearning')
            
            # Armazena os melhores episódios
            if astar_result:
                self.best_episodes['astar'].append({
                    'round': round_num,
                    'grid': grid,
                    'start': maze.start,
                    'goal': maze.goal,
                    'path': astar_result['path'],
                    'steps': astar_result['steps'],
                    'success_rate': success_rate,
                    'parameters': {
                        'width': result['width'],
                        'height': result['height'],
                        'branching': result['branching']
                    }
                })
            
            if qlearning_result and qlearning_result['success']:
                self.best_episodes['qlearning'].append({
                    'round': round_num,
                    'grid': grid,
                    'start': maze.start,
                    'goal': maze.goal,
                    'path': qlearning_result['path'],
                    'steps': qlearning_result['steps'],
                    'success_rate': success_rate,
                    'parameters': {
                        'width': result['width'],
                        'height': result['height'],
                        'branching': result['branching']
                    }
                })
    
    def _run_single_episode(self, maze: Maze, agent_type: str) -> Optional[Dict]:
        """Executa um único episódio com o labirinto gerado"""
        from Labirinto_adaptativo_improved import run_episode
        
        params = {
            'width': maze.width,
            'height': maze.height,
            'branching': maze.branching,
            'obstacle_density': maze.obstacle_density
        }
        
        if agent_type == 'astar':
            return run_episode(params, agent_type='astar', seed=42)
        else:
            # Para Q-Learning, cria um agente temporário
            q_agent = QLearningAgent(alpha=0.5, gamma=0.95, eps=0.1)
            result = run_episode(params, agent_type='qlearning', q_agent=q_agent, seed=42)
            return result
    
    def show_best_performances(self, agent_type: str = 'both', delay: int = 150):
        """Mostra as melhores performances animadas"""
        if agent_type in ['astar', 'both'] and self.best_episodes['astar']:
            print(f"\n🌟 MELHORES PERFORMANCES A* ({len(self.best_episodes['astar'])}):")
            for i, ep in enumerate(self.best_episodes['astar'], 1):
                params = ep['parameters']
                title = f"A* - Rodada {ep['round']} | {ep['steps']} passos | Taxa: {ep['success_rate']:.1%}"
                print(f"   {i}. {title}")
                print(f"      Parâmetros: {params['width']:.0f}x{params['height']:.0f}, branching={params['branching']:.3f}")
                
                if HAS_MATPLOTLIB:
                    animar_episodio(
                        ep['grid'], ep['start'], ep['goal'], 
                        ep['path'], title, delay
                    )
        
        if agent_type in ['qlearning', 'both'] and self.best_episodes['qlearning']:
            print(f"\n🤖 MELHORES PERFORMANCES Q-LEARNING ({len(self.best_episodes['qlearning'])}):")
            for i, ep in enumerate(self.best_episodes['qlearning'], 1):
                params = ep['parameters']
                title = f"Q-Learning - Rodada {ep['round']} | {ep['steps']} passos | Taxa: {ep['success_rate']:.1%}"
                print(f"   {i}. {title}")
                print(f"      Parâmetros: {params['width']:.0f}x{params['height']:.0f}, branching={params['branching']:.3f}")
                
                if HAS_MATPLOTLIB:
                    animar_episodio(
                        ep['grid'], ep['start'], ep['goal'], 
                        ep['path'], title, delay
                    )
    
    def create_performance_plots(self):
        """Cria gráficos de performance baseados nos dados"""
        if not self.results_data or not HAS_MATPLOTLIB:
            return
        
        print("\n📊 Criando gráficos de performance...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise de Performance - Sistema Adaptativo', fontsize=16, fontweight='bold')
        
        rounds = [r['round'] + 1 for r in self.results_data]
        
        # Gráfico 1: Taxa de sucesso
        ax1 = axes[0, 0]
        q_success = [r['q_success'] for r in self.results_data]
        astar_success = [r['astar_success'] for r in self.results_data]
        
        ax1.plot(rounds, q_success, 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Q-Learning')
        ax1.plot(rounds, astar_success, 's-', color='#2ecc71', linewidth=2, markersize=4, label='A* (100%)')
        ax1.set_title('Taxa de Sucesso por Rodada')
        ax1.set_xlabel('Rodada')
        ax1.set_ylabel('Taxa de Sucesso')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # Gráfico 2: Razão de eficiência
        ax2 = axes[0, 1]
        q_ratios = [r['q_median_ratio'] for r in self.results_data if r['q_median_ratio'] > 0]
        valid_rounds = rounds[:len(q_ratios)]
        
        ax2.plot(valid_rounds, q_ratios, '^-', color='#9b59b6', linewidth=2, markersize=4)
        ax2.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.7, label='Ótimo')
        ax2.set_title('Eficiência: Q-Learning vs A*')
        ax2.set_xlabel('Rodada')
        ax2.set_ylabel('Razão (Q/A*)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Epsilon
        ax3 = axes[1, 0]
        epsilons = [r['q_epsilon'] for r in self.results_data]
        ax3.plot(rounds, epsilons, 'd-', color='#f39c12', linewidth=2, markersize=4)
        ax3.set_title('Epsilon (Exploração)')
        ax3.set_xlabel('Rodada')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Evolução dos parâmetros
        ax4 = axes[1, 1]
        branchings = [r['branching'] for r in self.results_data]
        
        ax4.plot(rounds, branchings, 'o-', color='#3498db', linewidth=2, markersize=4)
        ax4.set_title('Evolução do Branching')
        ax4.set_xlabel('Rodada')
        ax4.set_ylabel('Branching')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_session(self):
        """Sessão interativa para explorar os resultados"""
        if not self.results_data:
            print("❌ Nenhum resultado para explorar")
            return
        
        print("\n🎮 SESSÃO INTERATIVA - VISUALIZADOR DE RESULTADOS")
        print("=" * 60)
        
        while True:
            print("\n📋 Opções disponíveis:")
            print("1. Análise de performance geral")
            print("2. Gerar melhores episódios para visualização")
            print("3. Mostrar melhores performances A*")
            print("4. Mostrar melhores performances Q-Learning")
            print("5. Mostrar ambos agentes")
            print("6. Criar gráficos de performance")
            print("7. Sair")
            
            choice = input("\nEscolha uma opção (1-7): ").strip()
            
            if choice == '1':
                self.analyze_performance()
            
            elif choice == '2':
                num_best = input("Número de episódios para gerar (padrão=3): ").strip()
                try:
                    num_best = int(num_best) if num_best else 3
                except ValueError:
                    num_best = 3
                self.generate_sample_episodes(num_best)
                print("✅ Episódios gerados! Use as opções 3-5 para visualizá-los.")
            
            elif choice in ['3', '4', '5']:
                delay = input("Velocidade da animação em ms (padrão=150): ").strip()
                try:
                    delay = int(delay) if delay else 150
                except ValueError:
                    delay = 150
                
                if choice == '3':
                    self.show_best_performances('astar', delay)
                elif choice == '4':
                    self.show_best_performances('qlearning', delay)
                else:
                    self.show_best_performances('both', delay)
            
            elif choice == '6':
                self.create_performance_plots()
            
            elif choice == '7':
                print("👋 Encerrando sessão...")
                break
            
            else:
                print("❌ Opção inválida!")


def main():
    """Função principal"""
    print("🎬 VISUALIZADOR DE RESULTADOS - SISTEMA ADAPTATIVO")
    print("=" * 60)
    
    # Verifica se existe arquivo de resultados
    csv_file = 'adaptive_results_improved.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Arquivo {csv_file} não encontrado!")
        print("\n💡 Execute primeiro o treinamento:")
        print("   python Labirinto_adaptativo.py")
        return
    
    # Cria visualizador
    visualizer = ResultsVisualizer(csv_file)
    
    # Menu principal
    print(f"\n📊 Resultados carregados: {len(visualizer.results_data)} rodadas")
    
    while True:
        print(f"\n🎯 Menu Principal:")
        print("1. Análise geral dos resultados")
        print("2. Sessão interativa completa")
        print("3. Gerar e mostrar melhores episódios")
        print("4. Sair")
        
        choice = input("\nEscolha uma opção (1-4): ").strip()
        
        if choice == '1':
            visualizer.analyze_performance()
        
        elif choice == '2':
            visualizer.interactive_session()
        
        elif choice == '3':
            print("\n🎬 Gerando melhores episódios...")
            visualizer.generate_sample_episodes(3)
            visualizer.show_best_performances('both')
        
        elif choice == '4':
            print("👋 Encerrando...")
            break
        
        else:
            print("❌ Opção inválida!")


if __name__ == "__main__":
    if not HAS_MATPLOTLIB:
        print("Este script requer matplotlib!")
        print("Instale com: pip install matplotlib")
    else:
        main()