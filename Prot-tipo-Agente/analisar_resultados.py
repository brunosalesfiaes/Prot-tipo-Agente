"""
Analisador de Resultados - Sistema Adaptativo de Labirintos
Script simples para visualizar resultados do treinamento
"""

import csv
import os
from typing import Dict, List

def carregar_resultados(arquivo_csv: str) -> List[Dict]:
    """Carrega resultados do arquivo CSV"""
    if not os.path.exists(arquivo_csv):
        print(f"❌ Arquivo {arquivo_csv} não encontrado!")
        print("Execute primeiro: python Labirinto_adaptativo.py")
        return []
    
    resultados = []
    with open(arquivo_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for linha in reader:
            # Converte valores numéricos
            for chave in linha:
                try:
                    linha[chave] = float(linha[chave])
                except ValueError:
                    pass  # Mantém como string se não conseguir converter
            resultados.append(linha)
    
    return resultados

def analisar_progresso(resultados: List[Dict]):
    """Analisa o progresso do treinamento"""
    if not resultados:
        return
    
    print("="*70)
    print("ANÁLISE DOS RESULTADOS - SISTEMA ADAPTATIVO DE LABIRINTOS")
    print("="*70)
    
    print(f"\n📊 Total de rodadas: {len(resultados)}")
    
    # Primeira e última rodada
    primeira = resultados[0]
    ultima = resultados[-1]
    
    print(f"\n📈 EVOLUÇÃO:")
    print(f"   Parâmetros iniciais: {primeira['width']:.0f}x{primeira['height']:.0f}, branching={primeira['branching']:.3f}")
    print(f"   Parâmetros finais:   {ultima['width']:.0f}x{ultima['height']:.0f}, branching={ultima['branching']:.3f}")
    
    # Taxa de sucesso do Q-Learning
    sucessos_q = [r['q_success'] for r in resultados]
    sucessos_astar = [r['astar_success'] for r in resultados]
    
    print(f"\n🎯 TAXA DE SUCESSO:")
    print(f"   Q-Learning inicial: {sucessos_q[0]:.1%}")
    print(f"   Q-Learning final:   {sucessos_q[-1]:.1%}")
    print(f"   Melhor Q-Learning:  {max(sucessos_q):.1%}")
    print(f"   A* (oráculo):       {sucessos_astar[0]:.1%} (sempre 100%)")
    
    # Razão de eficiência
    razoes_q = [r['q_median_ratio'] for r in resultados if r['q_median_ratio'] > 0]
    if razoes_q:
        print(f"\n⚡ EFICIÊNCIA (Passos Q-Learning / Passos Ótimos):")
        print(f"   Razão inicial: {razoes_q[0]:.2f}x")
        print(f"   Razão final:   {razoes_q[-1]:.2f}x")
        print(f"   Melhor razão:  {min(razoes_q):.2f}x")
    
    # Epsilon (exploração)
    epsilons = [r['q_epsilon'] for r in resultados]
    print(f"\n🔍 EXPLORAÇÃO (Epsilon):")
    print(f"   Inicial: {epsilons[0]:.3f}")
    print(f"   Final:   {epsilons[-1]:.3f}")
    print(f"   Episódios treinados: {ultima['q_episodes_trained']:.0f}")
    
    # Estatísticas por quartis
    n = len(resultados)
    quartis = [n//4, n//2, 3*n//4]
    indices = ["25%", "50%", "75%", "100%"]
    
    print(f"\n📊 PROGRESSO POR QUARTIS:")
    for i, (q_idx, idx_name) in enumerate(zip([0] + quartis, indices)):
        r = resultados[q_idx]
        print(f"   {idx_name:4s}: Taxa={r['q_success']:.1%}, "
              f"Razão={r['q_median_ratio']:.2f if r['q_median_ratio'] > 0 else 'N/A':>4s}, "
              f"Epsilon={r['q_epsilon']:.3f}")

def criar_grafico_simples(resultados: List[Dict]):
    """Cria gráfico simples usando ASCII (se matplotlib disponível)"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Análise do Treinamento Adaptativo', fontsize=14, fontweight='bold')
        
        rodadas = [r['round'] + 1 for r in resultados]
        
        # Taxa de sucesso
        sucessos_q = [r['q_success'] for r in resultados]
        ax1.plot(rodadas, sucessos_q, 'b-o', linewidth=2, markersize=4)
        ax1.set_title('Taxa de Sucesso Q-Learning')
        ax1.set_xlabel('Rodada')
        ax1.set_ylabel('Taxa de Sucesso')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3)
        
        # Razão de eficiência
        razoes = [r['q_median_ratio'] for r in resultados if r['q_median_ratio'] > 0]
        rodadas_ratio = rodadas[:len(razoes)]
        ax2.plot(rodadas_ratio, razoes, 'r-s', linewidth=2, markersize=4)
        ax2.set_title('Eficiência Relativa')
        ax2.set_xlabel('Rodada')
        ax2.set_ylabel('Q-Learning / A*')
        ax2.grid(True, alpha=0.3)
        
        # Epsilon
        epsilons = [r['q_epsilon'] for r in resultados]
        ax3.plot(rodadas, epsilons, 'g-^', linewidth=2, markersize=4)
        ax3.set_title('Epsilon (Exploração)')
        ax3.set_xlabel('Rodada')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)
        
        # Branching
        branchings = [r['branching'] for r in resultados]
        ax4.plot(rodadas, branchings, 'm-d', linewidth=2, markersize=4)
        ax4.set_title('Parâmetro Branching')
        ax4.set_xlabel('Rodada')
        ax4.set_ylabel('Branching')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n📈 Gráfico salvo com matplotlib!")
        
    except ImportError:
        print("\n📊 Matplotlib não disponível. Execute: pip install matplotlib")
        print("   Mas você pode ver a análise textual acima.")

def main():
    """Função principal"""
    arquivo = 'adaptive_results_improved.csv'
    
    print("🔍 Carregando resultados...")
    resultados = carregar_resultados(arquivo)
    
    if resultados:
        analisar_progresso(resultados)
        criar_grafico_simples(resultados)
        
        print("\n" + "="*70)
        print("✅ Análise concluída!")
        print("="*70)
    else:
        print("\n❌ Não foi possível carregar os resultados.")
        print("   Execute o treinamento primeiro: python Labirinto_adaptativo.py")

if __name__ == "__main__":
    main()