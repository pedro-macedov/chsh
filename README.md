# chsh
Repositório associado à dissertação **"O Uso de Redes Neurais Artificiais na Análise de Não Localidade em Cenários Relaxados"**, desenvolvida no Programa de Pós-graduação em Física Aplicada, vinculado à Universidade Federal Rural de Pernambuco.

Esse repositório contém:
- um script para a obtenção das facetas do politopo local do cenário CHSH;  
- a implementação de um "oráculo" de não localidade para o cenário CHSH padrão, sem relaxação;
- a implementação de um "oráculo" de não localidade para o cenário CHSH com relaxação unilateral;
- a implementação de um "oráculo" de não localidade para o cenário CHSH com relaxação superluminal;
- a implementação de um "oráculo" de não localidade para o cenário CHSH com relaxação causal.

### 1. script para a obtenção das facetas do politopo local do cenário CHSH

O script encontra-se no repositório nomeado como "facetas_polymake.py" e foi mencionado na seção 2.2.5 da dissertação. Para executá-lo é necessário fazer o download do software polymake, que pode ser realizado através do site https://polymake.org/doku.php/download/start, e sua respectiva instalação.

### 2. Implementações de "oráculos" de nao localidade para o cenário CHSH
Os códigos de cada uma das implementações de "oráculos" de não localidade estão divididas em 4 arquivos:
- architecture.py: contém a arquitetura da rede neural, o forward da rede e a função que realiza a inicialização dos pârametros;
- data_generation.py: contém as funções que geram os dados de entrada da rede e também a função que gera a distribuição da PR-box adicionada de ruído branco;
- train.py: contém as funções referentes ao processo de treino da rede neural;
- main.py: arquivo a ser executado. Chama as funções contidas nos outros arquivos, gerando os dados, treinando a rede e, ao fim, gerando os gráficos.
  #### Padrão, sem relaxação
  Os códigos dessa implemementação estão presentes na pasta "Padrão" deste repositório. O detalhamento do seu desenvolvimento está presente na seção 5.2.1 da dissertação, e os resultados obtidos estão presentes no início do capítulo 6.
  #### Relaxação unilateral
  Os códigos dessa implemementação estão presentes no caminho "Relaxações\Retrocausal unilateral" deste repositório. O detalhamento do seu desenvolvimento está presente na seção 5.2.2 da dissertação, e os resultados obtidos estão presentes na seção 6.1.
  #### Relaxação superluminal
  Os códigos dessa implemementação estão presentes no caminho "Relaxações\Superluminal" deste repositório. O detalhamento do seu desenvolvimento está presente na seção 5.2.2 da dissertação, e os resultados obtidos estão presentes na seção 6.1.
  #### Relaxação causal
  Os códigos dessa implemementação estão presentes no caminho "Relaxações\Causal" deste repositório. O detalhamento do seu desenvolvimento está presente na seção 5.2.3 da dissertação, e os resultados obtidos estão presentes na seção 5.2.

## Dependências

O projeto utiliza Python e as seguintes bibliotecas principais:

- PyTorch
- NumPy
- Matplotlib
- SciPy

Para instalar as dependências:

```bash
pip install -r requirements.txt
