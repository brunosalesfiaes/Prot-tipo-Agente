# Sistema Adaptativo de Labirintos

Este repositório implementa um sistema experimental de agentes em labirintos adaptativos. O objetivo é comparar um oráculo (A*) com um agente aprendiz (Q-Learning) enquanto um controlador ajusta a dificuldade dos labirintos.

## Visão Geral
- Geração procedural de labirintos (algoritmo tipo *recursive backtracker*).
- Oráculo A* para solução ótima dos labirintos.
- Agente Q-Learning com política ε-greedy e decaimento de ε ao longo do treinamento.
- Controlador adaptativo de dificuldade que ajusta parâmetros dos labirintos conforme desempenho.
- Registro de resultados em CSV para análise e visualização posterior.

## Principais arquivos
- `Labirinto_adaptativo.py` — Implementação principal do ambiente, agente Q-Learning, oráculo A* e loop de treinamento.
- `Labirinto_adaptativo_improved.py` — Versão alternativa/compatível com pequenas melhorias e correções.
- `Visualizador_Treinamento_Aprimorado.py` — Visualizador com treino em tempo real e animações das melhores performances.
- `Visualizador_Resultados.py` — Visualizador / ferramentas de plot para análise dos CSVs gerados.
- `analisar_resultados.py` — Scripts para análise textual e estatística dos resultados (gera métricas a partir do CSV).
- `adaptive_results_improved.csv` — Exemplo/resultado gerado pelo treinamento.
- `demonstracao_rapida.py` — Execução rápida/demo do sistema.
- `teste_correcoes.py`, `teste_visualizadores.py` — Scripts de teste e verificação.
- `launcher.py` — Launcher unificado para executar visualizadores e utilitários.

## Dependências
- Python 3.8+
- `matplotlib` (visualização)
- `numpy` (opcional, usado para estatísticas; há fallback quando ausente)

Instalação (PowerShell):
```powershell
pip install matplotlib numpy
```

## Como executar
- Iniciar o launcher (menu interativo):
```powershell
python launcher.py
```
- Treinamento completo:
```powershell
python Labirinto_adaptativo.py
```
- Visualizar resultados:
```powershell
python Visualizador_Resultados.py
```
- Demonstração rápida:
```powershell
python demonstracao_rapida.py
```

## Epsilon decay (decaimento de ε)

O agente Q-Learning usa uma política ε-greedy para balancear exploração/exploração. O decaimento de ε reduz gradualmente a taxa de exploração ao longo dos episódios.

Esquema recomendado (exponencial por episódio):

```python
eps_start = 1.0
eps_min = 0.05
decay_rate = 0.005  # ajustar conforme número de episódios

epsilon = eps_min + (eps_start - eps_min) * math.exp(-decay_rate * episode)
```

Dicas:
- Valores comuns: `eps_start = 1.0`, `eps_min = 0.01–0.1`.
- Ajuste `decay_rate` ao número total de episódios: mais episódios → menor `decay_rate`.
- Evite decair muito rápido para não convergir prematuramente para policy ruim.

## Estrutura de implementação (resumo técnico)
- Maze / Ambiente: representação matricial/textual com métodos de geração e utilitários `reset()` / `step()`.
- Oráculo: algoritmo A* que fornece caminhos ótimos — usado para comparação e métricas.
- Agente: Q-Learning com tabela Q, atualização por passo, política ε-greedy e decaimento de ε por episódio.
- Controlador de dificuldade: ajusta parâmetros do gerador (ramificações / tamanho) conforme métricas de desempenho (por exemplo, taxa de sucesso ou tempo médio).
- Persistência: resultados salvos em CSV para análises posteriores pelos visualizadores.

## Logs & Análise
- Os resultados de treinamento podem ser encontrados em `adaptive_results_improved.csv`.
- Use `analisar_resultados.py` para métricas e `Visualizador_Resultados.py` para gráficos.
---
