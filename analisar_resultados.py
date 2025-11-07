"""
Script para analisar e visualizar os resultados do sistema adaptativo de labirintos.
Gera gráficos simples usando matplotlib (se disponível) ou apenas estatísticas em texto.
"""

import csv
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib não disponível. Apenas estatísticas em texto serão geradas.")


def ler_log_csv(arquivo='adaptive_log.csv'):
    """Lê o arquivo CSV de log e retorna os dados."""
    if not os.path.exists(arquivo):
        print(f"Arquivo {arquivo} não encontrado. Execute primeiro adaptive_maze.py")
        return None
    
    dados = []
    with open(arquivo, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dados.append({
                'round': int(row['round']),
                'width': int(row['width']),
                'height': int(row['height']),
                'branching': float(row['branching']),
                'astar_success': float(row['astar_success']),
                'astar_median_steps': float(row['astar_median_steps']),
                'q_success': float(row['q_success']),
                'q_median_steps': float(row['q_median_steps'])
            })
    return dados


def analisar_estatisticas(dados):
    """Calcula estatísticas descritivas dos dados."""
    if not dados:
        return
    
    print("\n" + "="*60)
    print("ANÁLISE DE RESULTADOS - SISTEMA ADAPTATIVO DE LABIRINTOS")
    print("="*60)
    
    # Estatísticas gerais
    rounds = [d['round'] for d in dados]
    branching = [d['branching'] for d in dados]
    astar_steps = [d['astar_median_steps'] for d in dados]
    q_steps = [d['q_median_steps'] for d in dados]
    astar_success = [d['astar_success'] for d in dados]
    q_success = [d['q_success'] for d in dados]
    
    print(f"\nTotal de rounds: {len(dados)}")
    print(f"\n--- Parâmetros do Labirinto ---")
    print(f"Branching - Mín: {min(branching):.4f}, Máx: {max(branching):.4f}, "
          f"Média: {sum(branching)/len(branching):.4f}")
    print(f"Tamanho - Mín: {min([d['width'] for d in dados])}x{min([d['height'] for d in dados])}, "
          f"Máx: {max([d['width'] for d in dados])}x{max([d['height'] for d in dados])}")
    
    print(f"\n--- Desempenho do Agente A* (Ótimo) ---")
    print(f"Taxa de sucesso média: {sum(astar_success)/len(astar_success):.2%}")
    print(f"Passos médios: {sum(astar_steps)/len(astar_steps):.1f}")
    print(f"Passos - Mín: {min(astar_steps):.1f}, Máx: {max(astar_steps):.1f}")
    
    print(f"\n--- Desempenho do Agente Q-Learning ---")
    print(f"Taxa de sucesso média: {sum(q_success)/len(q_success):.2%}")
    print(f"Passos médios: {sum(q_steps)/len(q_steps):.1f}")
    print(f"Passos - Mín: {min(q_steps):.1f}, Máx: {max(q_steps):.1f}")
    
    # Análise de adaptatividade
    print(f"\n--- Análise de Adaptatividade ---")
    if len(dados) > 1:
        branching_inicial = branching[0]
        branching_final = branching[-1]
        mudanca = ((branching_final - branching_inicial) / branching_inicial) * 100 if branching_inicial > 0 else 0
        print(f"Mudança no branching: {mudanca:+.1f}%")
        
        # Verifica se houve estabilização
        ultimos_10 = branching[-10:] if len(branching) >= 10 else branching
        variacao_final = max(ultimos_10) - min(ultimos_10)
        print(f"Variação nos últimos rounds: {variacao_final:.4f}")
        if variacao_final < 0.05:
            print("[OK] Sistema parece ter estabilizado!")
        else:
            print(">> Sistema ainda esta ajustando parametros")
    
    print("\n" + "="*60)


def plotar_graficos(dados):
    """Gera gráficos de visualização dos resultados."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib não disponível. Instale com: pip install matplotlib")
        return
    
    if not dados:
        return
    
    rounds = [d['round'] for d in dados]
    branching = [d['branching'] for d in dados]
    astar_steps = [d['astar_median_steps'] for d in dados]
    q_steps = [d['q_median_steps'] for d in dados]
    astar_success = [d['astar_success'] for d in dados]
    q_success = [d['q_success'] for d in dados]
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Análise do Sistema Adaptativo de Labirintos', fontsize=14, fontweight='bold')
    
    # Gráfico 1: Branching ao longo do tempo
    axes[0, 0].plot(rounds, branching, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Branching')
    axes[0, 0].set_title('Evolução do Parâmetro Branching')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Passos médios (A* vs Q-Learning)
    axes[0, 1].plot(rounds, astar_steps, 'g-', linewidth=2, marker='s', markersize=4, label='A* (Ótimo)')
    axes[0, 1].plot(rounds, q_steps, 'r-', linewidth=2, marker='^', markersize=4, label='Q-Learning')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Passos Médios')
    axes[0, 1].set_title('Desempenho: Passos para Resolver')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Taxa de sucesso
    axes[1, 0].plot(rounds, astar_success, 'g-', linewidth=2, marker='s', markersize=4, label='A*')
    axes[1, 0].plot(rounds, q_success, 'r-', linewidth=2, marker='^', markersize=4, label='Q-Learning')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Taxa de Sucesso')
    axes[1, 0].set_title('Taxa de Sucesso ao Longo do Tempo')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Relação Branching vs Desempenho
    axes[1, 1].scatter(branching, q_steps, alpha=0.6, s=50, c=rounds, cmap='viridis')
    axes[1, 1].set_xlabel('Branching')
    axes[1, 1].set_ylabel('Passos (Q-Learning)')
    axes[1, 1].set_title('Relação: Dificuldade vs Desempenho')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Round')
    
    plt.tight_layout()
    
    # Salvar figura
    output_file = 'analise_resultados.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Graficos salvos em: {output_file}")
    
    # Mostrar gráficos
    plt.show()


def main():
    """Função principal."""
    dados = ler_log_csv()
    
    if dados:
        analisar_estatisticas(dados)
        
        if HAS_MATPLOTLIB:
            resposta = input("\nDeseja gerar gráficos? (s/n): ").strip().lower()
            if resposta == 's':
                plotar_graficos(dados)
        else:
            print("\nPara gerar gráficos, instale matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()

