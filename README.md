# Prot-tipo-Agente
Protótipo do trabalho A3 UNIFACS, "Agente labirinto" 

## 📋 Requisitos

- Python 3.8 ou superior
- Bibliotecas: `numpy` (recomendado), `matplotlib` (opcional, para visualização)

Instalação das dependências:
```bash
pip install -r requirements.txt
```

## 🎯 Sistema Adaptativo de Labirintos

Este projeto implementa um sistema completo de **geração procedural de labirintos com ajuste adaptativo de dificuldade**, incluindo:

### Componentes Principais

1. **Gerador de Labirintos (`Maze`)**
   - Geração procedural usando algoritmo Recursive Backtracker
   - Parâmetros ajustáveis: tamanho, branching (bifurcações/loops), densidade de obstáculos
   - Garante conectividade entre início e objetivo

2. **Agentes Inteligentes**
   - **A* Agent**: Planejador ótimo (heurística Manhattan) - serve como medidor de dificuldade
   - **Q-Learning Agent**: Agente de aprendizado por reforço tabular para testar adaptatividade

3. **Controlador Adaptativo (`DifficultyController`)**
   - Ajusta dinamicamente parâmetros do labirinto baseado em métricas de desempenho
   - Regras heurísticas para aumentar/reduzir dificuldade
   - Mantém histórico de ajustes

4. **Sistema de Avaliação**
   - Métricas: taxa de sucesso, passos médios, razão de eficiência
   - Logs em CSV para análise posterior
   - Visualização de resultados (com matplotlib)

### 🚀 Como Usar

#### Executar o Sistema Adaptativo

```bash
python adaptive_maze.py
```

Este script executa:
- 30 rounds de avaliação
- Alterna entre medição com A* e treinamento do agente Q-Learning
- Gera arquivo `adaptive_log.csv` com resultados

#### Analisar Resultados

```bash
python analisar_resultados.py
```

Este script:
- Lê o arquivo `adaptive_log.csv`
- Exibe estatísticas descritivas
- Gera gráficos de evolução (se matplotlib estiver instalado)

### 📊 Métricas Coletadas

- **Taxa de sucesso**: Percentual de episódios resolvidos
- **Passos médios**: Número médio de passos para resolver
- **Razão de eficiência**: `passos_reais / caminho_mínimo` (quanto mais próximo de 1.0, melhor)
- **Parâmetros adaptados**: Branching, tamanho do labirinto

### 🔬 Questão de Pesquisa

**"É possível ajustar dinamicamente a dificuldade de um labirinto baseado no desempenho do agente?"**

O sistema demonstra que sim, através de:
- Medição contínua do desempenho (A* como oráculo)
- Ajuste reativo dos parâmetros de geração
- Estabilização gradual da dificuldade

### 📁 Estrutura de Arquivos

- `adaptive_maze.py`: Sistema principal (geração, agentes, controlador)
- `analisar_resultados.py`: Script de análise e visualização
- `labirintoPy.py`: Implementação original com BFS e visualização
- `NovoLabirinto/Labirinto.py`: Arquivo em desenvolvimento
- `requirements.txt`: Dependências do projeto

### 🎨 Visualização

O sistema original (`labirintoPy.py`) inclui visualização animada com matplotlib:
```bash
python labirintoPy.py
```

### 📝 Próximas Melhorias

- [ ] Controlador mais sofisticado (PID, Bandits, Bayesian Optimization)
- [ ] Agente RL mais avançado (DQN com PyTorch)
- [ ] Métricas cognitivas adicionais (decisões críticas, entropia de trajetórias)
- [ ] Geradores condicionais via ML (VAE/GAN)
- [ ] Framework multiagente com memória e armazenamento

### 🔗 Referências

- Frameworks Python para Agentes de IA: https://blog.dsacademy.com.br/8-principais-frameworks-python-para-agentes-de-ia/
