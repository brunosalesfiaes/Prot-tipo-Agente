#!/usr/bin/env python3
"""
🚀 LAUNCHER UNIFICADO - SISTEMA ADAPTATIVO DE LABIRINTOS
Centraliza o acesso a todos os visualizadores e funcionalidades
"""

import os
import sys

def mostrar_banner():
    """Mostra banner do sistema"""
    print("=" * 80)
    print("🎯 SISTEMA ADAPTATIVO DE LABIRINTOS - LAUNCHER UNIFICADO")
    print("=" * 80)
    print()

def verificar_dependencias():
    """Verifica se as dependências estão instaladas"""
    dependencias_ok = True
    
    try:
        import matplotlib
        print("✅ matplotlib - OK")
    except ImportError:
        print("❌ matplotlib - NÃO INSTALADO")
        dependencias_ok = False
    
    try:
        import numpy
        print("✅ numpy - OK")
    except ImportError:
        print("⚠️  numpy - Opcional (será usado fallback)")
    
    return dependencias_ok

def mostrar_menu_principal():
    """Mostra o menu principal"""
    print("\n📋 FUNCIONALIDADES DISPONÍVEIS:")
    print()
    print("🎯 TREINAMENTO E VISUALIZAÇÃO:")
    print("   1.  Visualizador de Treinamento Aprimorado (Novo)")
    print("       • Treinamento em tempo real")
    print("       • Seleção automática das melhores rodadas")
    print("       • Animações das performances top")
    print()
    print("   2.  Visualizador de Resultados (Análise)")
    print("       • Análise de treinamentos anteriores")
    print("       • Gráficos de evolução")
    print("       • Recriação das melhores performances")
    print()
    print("🔧 SISTEMA BASE:")
    print("   3.  Sistema Principal (Treinamento Completo)")
    print("       • 25 rodadas de treinamento adaptativo")
    print("       • Geração de arquivo CSV com resultados")
    print()
    print("📊 ANÁLISE E TESTES:")
    print("   4.  Analisador de Resultados (CSV)")
    print("       • Análise textual dos resultados")
    print("       • Estatísticas e métricas")
    print()
    print("   5.  Teste dos Visualizadores")
    print("       • Verificação de funcionalidades")
    print("       • Demonstração das capacidades")
    print()
    print("🎬 UTILITÁRIOS:")
    print("   6.  Demonstração Rápida")
    print("       • Teste básico do sistema")
    print()
    print("   0.  Sair")
    print()

def executar_opcao(opcao):
    """Executa a opção escolhida"""
    
    if opcao == "1":
        print("\n🚀 Iniciando Visualizador de Treinamento Aprimorado...")
        print("=" * 60)
        os.system("python Visualizador_Treinamento_Aprimorado.py")
        
    elif opcao == "2":
        print("\n📊 Iniciando Visualizador de Resultados...")
        print("=" * 60)
        os.system("python Visualizador_Resultados.py")
        
    elif opcao == "3":
        print("\n🎯 Iniciando Sistema Principal...")
        print("=" * 60)
        os.system("python Labirinto_adaptativo.py")
        
    elif opcao == "4":
        print("\n📈 Iniciando Analisador de Resultados...")
        print("=" * 60)
        os.system("python analisar_resultados.py")
        
    elif opcao == "5":
        print("\n🧪 Iniciando Teste dos Visualizadores...")
        print("=" * 60)
        os.system("python teste_visualizadores.py")
        
    elif opcao == "6":
        print("\n⚡ Iniciando Demonstração Rápida...")
        print("=" * 60)
        os.system("python demonstracao_rapida.py")
        
    elif opcao == "0":
        print("\n👋 Encerrando launcher...")
        return False
        
    else:
        print("\n❌ Opção inválida!")
        return True
    
    input("\n⏸️  Pressione ENTER para continuar...")
    return True

def mostrar_info_sistema():
    """Mostra informações do sistema"""
    print("\n📁 ARQUIVOS PRINCIPAIS:")
    print()
    
    arquivos = {
        "Labirinto_adaptativo.py": "Sistema principal corrigido",
        "Labirinto_adaptativo_improved.py": "Cópia para compatibilidade",
        "Visualizador_Treinamento_Aprimorado.py": "Visualizador com treinamento",
        "Visualizador_Resultados.py": "Visualizador de resultados",
        "analisar_resultados.py": "Analisador textual",
        "teste_visualizadores.py": "Teste das funcionalidades",
        "demonstracao_rapida.py": "Demonstração básica"
    }
    
    for arquivo, descricao in arquivos.items():
        if os.path.exists(arquivo):
            print(f"   ✅ {arquivo:35s} - {descricao}")
        else:
            print(f"   ❌ {arquivo:35s} - {descricao}")
    
    print(f"\n📄 DOCUMENTAÇÃO:")
    print("   • GUIA_VISUALIZADORES.md - Guia completo de uso")
    print("   • README_CORRECOES.md - Correções aplicadas")
    print("   • RELATORIO_CORRECOES.md - Relatório detalhado")

def main():
    """Função principal do launcher"""
    mostrar_banner()
    
    # Verifica dependências
    print("🔍 Verificando dependências...")
    deps_ok = verificar_dependencias()
    
    if not deps_ok:
        print("\n❌ Algumas dependências estão faltando!")
        print("💡 Execute: pip install matplotlib")
        resposta = input("\nContinuar mesmo assim? (s/N): ").lower()
        if resposta not in ['s', 'sim', 'y', 'yes']:
            return
    
    print(f"\n📊 Status do Sistema:")
    mostrar_info_sistema()
    
    # Loop principal
    while True:
        mostrar_menu_principal()
        
        opcao = input("🎯 Escolha uma opção (0-6): ").strip()
        
        if not executar_opcao(opcao):
            break
        
        # Limpa a tela (opcional)
        # os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_resumo_final():
    """Mostra resumo final ao sair"""
    print("\n" + "=" * 80)
    print("🎉 SISTEMA ADAPTATIVO DE LABIRINTOS")
    print("=" * 80)
    print()
    print("✅ VISUALIZADORES APRIMORADOS IMPLEMENTADOS:")
    print("   • Treinamento em tempo real com seleção das melhores rodadas")
    print("   • Modelo de animação antigo integrado e funcional")
    print("   • Análise completa de resultados com gráficos")
    print("   • Comparação entre episódios A* vs Q-Learning")
    print()
    print("🎯 FUNCIONALIDADES PRINCIPAIS:")
    print("   ✅ Correção de erros de formatação None")
    print("   ✅ Resolução de problemas de importação")
    print("   ✅ Integração do modelo de animação sofisticado")
    print("   ✅ Sistema de seleção das melhores performances")
    print("   ✅ Interface interativa avançada")
    print()
    print("🚀 COMANDOS PRINCIPAIS:")
    print("   python launcher.py                    # Menu principal")
    print("   python Visualizador_Treinamento_Aprimorado.py  # Treinamento")
    print("   python Visualizador_Resultados.py     # Análise de resultados")
    print()
    print("📖 DOCUMENTAÇÃO:")
    print("   • GUIA_VISUALIZADORES.md - Como usar os visualizadores")
    print("   • README_CORRECOES.md - Resumo das correções")
    print()
    print("👋 O sistema está pronto para uso completo!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
    finally:
        mostrar_resumo_final()