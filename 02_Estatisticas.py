# ==================== Ativar VENV ==================== #
# python -m venv venv -> Cria novo
# .\venv\Scripts\activate -> Ativa
# deactivate -> Desativa
# python -m venv venv --clear -> Recriar do zero
# Remove-Item -Recurse -Force venv -> Remove

# =============================================== #
# ================ INTEGRAÇÕES ================= #
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================== #
# =============== CONFIG INICIAL =============== #
# Configuração inicial
ARQUIVO_CSV = 'vehicles.csv'
ARQUIVO_FEATHER = 'vehicles.feather'

print(f"\nRafael, iniciando processamento do arquivo {ARQUIVO_CSV}...")

# Verifica e carrega os dados
if not Path(ARQUIVO_FEATHER).exists():
    print("Arquivo feather não encontrado. Criando a partir do CSV...")
    try:
        df = pd.read_csv(ARQUIVO_CSV)
        df.to_feather(ARQUIVO_FEATHER)
        print("✅ Arquivo feather criado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao processar arquivo CSV: {e}")
        exit()
else:
    print("✅ Arquivo feather encontrado. Carregando...")

try:
    df = pd.read_feather(ARQUIVO_FEATHER)
    print(f"\nDataset carregado com {len(df):,} registros e {len(df.columns)} colunas.")
except Exception as e:
    print(f"❌ Erro ao carregar arquivo feather: {e}")
    exit()
# =============================================== #
# ================== FUNÇÕES =================== #
def estatisticas_colunas(df):
    
    """
    Retorna estatísticas para colunas numéricas (int/float)
    com formatação legível
    """
    numericas = df.select_dtypes(include=['int64', 'float64'])
    
    if numericas.empty:
        print("\n⚠️ Nenhuma coluna numérica encontrada!")
        return
    
    estatisticas = pd.DataFrame({
        'Coluna': numericas.columns,
        'Tipo': numericas.dtypes.values,
        'Média': numericas.mean(),
        'Mediana': numericas.median(),
        'Desvio Padrão': numericas.std(),
        'Mínimo': numericas.min(),
        'Máximo': numericas.max(),
        'Nulos (%)': (numericas.isnull().mean() * 100).round(2)
    })
    
    # Configura formatação
    pd.options.display.float_format = '{:,.2f}'.format
    
    print("\n" + "="*50)
    print("📈 ESTATÍSTICAS DAS COLUNAS NUMÉRICAS:")
    print(estatisticas.to_string(index=False))
    print("="*50)
    
    pd.reset_option('display.float_format')

def comparacao_campos(df):
    """Analisa a relação entre dois campos com seleção de método estatístico"""
    print("\n" + "="*50)
    print("📌 COLUNAS DISPONÍVEIS PARA COMPARAÇÃO:")
    
    for i, coluna in enumerate(df.columns, 1):
        print(f"{i}. {coluna}")
        
     # Configuração inicial padrão
    metodo_nome = 'pearson'  # Definindo valor padrão
    metodo_desc = "📈 Análise de relação LINEAR (Pearson)"
    campo1 = None
    campo2 = None

    # Seleção dos campos
    print("\n" + "="*50)
    print("🔎 SELECIONE 2 CAMPOS PARA COMPARAÇÃO")
    
    while True:
        try:
            campo1_idx = int(input("▶ Digite o número do primeiro campo: ")) - 1
            campo2_idx = int(input("▶ Digite o número do segundo campo: ")) - 1
            
            if 0 <= campo1_idx < len(df.columns) and 0 <= campo2_idx < len(df.columns):
                campo1 = df.columns[campo1_idx]
                campo2 = df.columns[campo2_idx]
                break
            else:
                print("❌ Número inválido. Tente novamente.")
        except ValueError:
            print("❌ Por favor, digite apenas números.")

    # Configuração inicial para métodos
    metodo_nome = 'pearson'
    metodo_desc = "📈 Análise de relação LINEAR (Pearson)"
    
    # Seleção do coeficiente para variáveis numéricas
    if all(pd.api.types.is_numeric_dtype(df[c]) for c in [campo1, campo2]):
        print("\n" + "="*50)
        print("🧮 MÉTODO DE ANÁLISE ESTATÍSTICA:")
        print("1. Pearson [Padrão]")
        print(f"   • Covariância: {df[[campo1, campo2]].cov().iloc[0,1]:,.2f}")
        print("   • Interpretação covariância:")
        print("     - Valor positivo: As variáveis tendem a aumentar juntas")
        print("     - Valor negativo: Uma variável aumenta quando a outra diminui")
        print("     - Magnitude: Depende das escalas das variáveis")
        
        print("\n2. Spearman")
        print("   • Relações monotônicas (crescentes/decrescentes não-lineares)")
        
        print("\n3. Kendall")
        print("   • Concordância entre rankings (ideal para dados ordinais)")
        
        while True:
            try:
                escolha = input("\n▶ Digite o número do método (Enter para Pearson padrão): ")
                if escolha == "":
                    escolha = 1  # Default
                
                escolha = int(escolha)
                if escolha in [1, 2, 3]:
                    metodos = {
                        1: ('pearson', "📈 Análise de relação LINEAR (Pearson)"),
                        2: ('spearman', "🔄 Análise de tendência MONOTÔNICA (Spearman)"),
                        3: ('kendall', "🏷️ Análise de CONCORDÂNCIA (Kendall)")
                    }
                    metodo_nome, metodo_desc = metodos[escolha]
                    break
                else:
                    print("❌ Opção inválida. Digite 1, 2, 3 ou Enter para padrão")
            except ValueError:
                print("❌ Por favor, digite apenas números ou Enter")

    print("\n" + "="*50)
    print(f"{metodo_desc} entre '{campo1}' e '{campo2}':")

    # ANÁLISE NUMÉRICA x NUMÉRICA
    if all(pd.api.types.is_numeric_dtype(df[c]) for c in [campo1, campo2]):
        corr = df[[campo1, campo2]].corr(method=metodo_nome).iloc[0,1] 
        print(f"\n🔍 COEFICIENTE ({metodo_nome.upper()}): {corr:.2f}")
    
    # Leitura personalizada para cada método (VERSÃO COMPLETA E APRIMORADA)
    if metodo_nome == 'pearson':
        print(f"   Veja que de acordo com seu coeficiente de Pearson {corr:.2f}, a interpretação seria:")
        if abs(corr) >= 0.7:
            print(f"   'Forte correlação linear {'negativa' if corr < 0 else 'positiva'} entre as variáveis'")
            print(f"   • {campo1} e {campo2} variam {'inversamente' if corr < 0 else 'conjuntamente'} de forma previsível")
            print(f"   • {abs(corr)*100:.0f}% da variação pode ser explicada pela relação linear")
        elif abs(corr) >= 0.5:
            print(f"   'Correlação linear {'negativa' if corr < 0 else 'positiva'} moderada'") 
            print(f"   • Relação discernível, mas com alguns desvios")
            print(f"   • Quando {campo1} aumenta, {campo2} tende a {'diminuir' if corr < 0 else 'aumentar'}")
        elif abs(corr) >= 0.3:
            print(f"   'Correlação linear {'negativa' if corr < 0 else 'positiva'} fraca'")
            print(f"   • Alguma relação detectada, mas pouco confiável para previsões")
        else:
            print("   'Praticamente nenhuma correlação linear detectada'")
            print("   • As variáveis não mostram padrão linear mensurável")
            
    elif metodo_nome == 'spearman':
        print(f"   Veja que de acordo com seu coeficiente de Spearman {corr:.2f}, a interpretação seria:")
        if abs(corr) >= 0.7:
            print(f"   'Forte tendência monotônica {'decrescente' if corr < 0 else 'crescente'}'")
            print(f"   • Quando {campo1} aumenta, {campo2} {'sempre diminui' if corr < 0 else 'sempre aumenta'} na maioria absoluta dos casos")
        elif abs(corr) >= 0.5:
            print(f"   'Tendência {'inversa' if corr < 0 else 'direta'} moderada, porém consistente'") 
            print(f"   • Padrão claro de {'decréscimo' if corr < 0 else 'crescimento'} em ~{abs(corr)*100:.0f}% dos pares observados")
        elif abs(corr) >= 0.3:
            print(f"   'Tendência {'negativa' if corr < 0 else 'positiva'} fraca, mas perceptível'")
            print(f"   • Quando {campo1} AUMENTA, o {campo2} tende a {'DIMINUIR' if corr < 0 else 'AUMENTAR'}")
            print(f"   • Quando {campo1} DIMINUI, o {campo2} tende a {'AUMENTAR' if corr < 0 else 'DIMINUIR'}")
            print(f"   • Força: Valor absoluto {abs(corr):.2f} (entre 0.3 e 0.5) → ~{abs(corr)*100:.0f}% dos pares seguem esse padrão")
        else:
            print("   'Relação pouco significativa entre as variáveis'")
            print("   • Não há padrão direcional consistente")
            
    else:  # Kendall
        print(f"   Veja que de acordo com seu coeficiente de Kendall {corr:.2f}, a interpretação seria:")
        if abs(corr) >= 0.7:
            print(f"   'Concordância {'negativa' if corr < 0 else 'positiva'} quase perfeita entre rankings'")
            print(f"   • A ordem dos valores de {campo1} e {campo2} {'sempre inverte' if corr < 0 else 'quase sempre coincide'}")
        elif abs(corr) >= 0.5:
            print(f"   'Concordância {'negativa' if corr < 0 else 'positiva'} moderada'") 
            print(f"   • Em ~{abs(corr)*100:.0f}% dos casos, os rankings mantêm relação {'inversa' if corr < 0 else 'direta'}")
        elif abs(corr) >= 0.3:
            print(f"   'Concordância {'negativa' if corr < 0 else 'positiva'} fraca'")
            print("   • Alguma relação nos rankings, mas inconsistente")
        else:
            print("   'Concordância insignificante entre rankings'")
            print("   • A ordem dos valores parece aleatória")
    
              
    # Interpretação específica
    print("\n📌 INTERPRETAÇÃO:")
    if metodo_nome == 'pearson':
        print("1.00: Correlação perfeita positiva")
        print("0.50: Correlação moderada positiva")
        print("0.00: Nenhuma correlação")
        print("-0.50: Correlação moderada negativa")
        print("-1.00: Correlação perfeita negativa")
    elif metodo_nome == 'spearman':
        print("1.00: Tendência monotônica perfeita crescente")
        print("0.50: Tendência monotônica moderada crescente")
        print("0.00: Nenhuma tendência")
        print("-0.50: Tendência monotônica moderada decrescente")
        print("-1.00: Tendência monotônica perfeita decrescente")
    else:  # Kendall
        print("1.00: Concordância perfeita")
        print("0.50: Concordância moderada")
        print("0.00: Nenhuma concordância")
        print("-0.50: Discordância moderada")
        print("-1.00: Discordância perfeita")
        
        # Análise do valor específico
        print(f"\n🔎 NO SEU CASO (Coeficiente: {corr:.2f}):")
        if abs(corr) >= 0.8:
            print("• Relação MUITO FORTE entre as variáveis")
        elif abs(corr) >= 0.5:
            print("• Relação MODERADA entre as variáveis")
        elif abs(corr) >= 0.3:
            print("• Relação FRACA entre as variáveis")
        else:
            print("• Praticamente NENHUMA relação detectada")
        
        if corr > 0:
            print("• Relação POSITIVA (as variáveis mudam na mesma direção)")
        elif corr < 0:
            print("• Relação NEGATIVA (as variáveis mudam em direções opostas)")

        # Estatísticas formatadas corretamente
        print("\n📈 ESTATÍSTICAS DESCRITIVAS:")
        stats = df[[campo1, campo2]].describe()
        
        # Formatação para 2 casas decimais sem notação científica
        pd.options.display.float_format = '{:,.2f}'.format
        print(stats.to_string(float_format='{:,.2f}'.format))
        pd.reset_option('display.float_format')
"""
def plotar_boxplot(df):
    #Gera boxplot com opções de transformação logarítmica e zoom
    print("\n" + "="*50)
    print("📊 CONFIGURAÇÃO DO BOXPLOT")
    
    # 1. Seleção da coluna (existente)
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print("\n🔢 COLUNAS NUMÉRICAS DISPONÍVEIS:")
    for i, col in enumerate(colunas_numericas, 1):
        print(f"{i}. {col}")
    
    try:
        col_idx = int(input("\n▶ Escolha o número da coluna: ")) - 1
        coluna = colunas_numericas[col_idx]
    except (ValueError, IndexError):
        print("❌ Seleção inválida!")
        return

    # 2. Configurações básicas (existente)
    print("\n⚙️ CONFIGURAÇÕES BÁSICAS:")
    fator = float(input("▶ Fator do IQR (1.5 ou 3.0): ") or "1.5")
    orientacao = input("▶ Orientação (h/v): ").lower() or "v"
    # 2.3. Tamanho da figura (com limites seguros)
    while True:
        try:
            largura = float(input("▶ Largura da figura (1-20, padrão=10): ") or 10)
            altura = float(input("▶ Altura da figura (1-20, padrão=6): ") or 6)
            if 1 <= largura <= 20 and 1 <= altura <= 20:
                break
            print("⚠️ Valores devem estar entre 1 e 20")
        except ValueError:
            print("⚠️ Digite números válidos")
            largura, altura = 10, 6

    # 3. NOVA OPÇÃO: Tipo de visualização (MODIFICADO)
    print("\n🔍 MELHORIA DE VISUALIZAÇÃO:")
    print("1. Padrão (recomendado para dados uniformes)")
    print("2. Escala logarítmica (recomendado para outliers extremos)")
    print("3. Zoom na área interquartil (foco no núcleo dos dados)")
    print("4. Retirar outliers da visualização")  # NOVA OPÇÃO
    escolha_visualizacao = input("▶ Escolha (1/2/3/4): ") or "1"

    # Cálculos estatísticos (existente)
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1
    mediana = df[coluna].median()
    media = df[coluna].mean()
    limite_inferior = max(0, q1 - fator * iqr)
    limite_superior = q3 + fator * iqr

    # Plotagem (CORREÇÃO DA ORIENTAÇÃO + NOVA OPÇÃO)
    plt.figure(figsize=(largura, altura), tight_layout=True)
    sns.set_style("whitegrid")
    
    # Configura orientação (CORRIGIDO)
    showfliers = escolha_visualizacao != "4"  # NOVO: Controla outliers
    if orientacao == "h":  # CORREÇÃO: 'h' agora funciona corretamente
        ax = sns.boxplot(y=df[coluna], whis=fator, color="skyblue", showfliers=showfliers)
    else:
        ax = sns.boxplot(x=df[coluna], whis=fator, color="skyblue", showfliers=showfliers)

    # Aplica a melhoria escolhida (existente)
    if escolha_visualizacao == "2":
        if orientacao == "h":
            plt.yscale('log')
            plt.ylabel(f"{coluna} (escala logarítmica)")
        else:
            plt.xscale('log')
            plt.xlabel(f"{coluna} (escala logarítmica)")
        titulo = f"Boxplot LOGARÍTMICO de '{coluna}'"
        
    elif escolha_visualizacao == "3":
        limite_superior_zoom = q3 + 5 * iqr
        if orientacao == "h":
            plt.ylim(0, limite_superior_zoom)
        else:
            plt.xlim(0, limite_superior_zoom)
        titulo = f"Boxplot de '{coluna}' (Zoom IQR)"
        
    else:
        titulo = f"Boxplot de '{coluna}'"

    # Linhas de referência
    linewidth = 2.5 if escolha_visualizacao != "3" else 3.0
    if orientacao == "h":
        plt.axhline(limite_superior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Sup: {limite_superior:,.2f}')
        plt.axhline(q3, color='green', linestyle='--', linewidth=linewidth, label=f'Q3: {q3:,.2f}')
        plt.axhline(media, color='purple', linestyle=':', linewidth=linewidth+0.5, label=f'Média: {media:,.2f}')
        plt.axhline(mediana, color='orange', linestyle='-', linewidth=linewidth+0.5, label=f'Mediana: {mediana:,.2f}')
        plt.axhline(q1, color='red', linestyle='--', linewidth=linewidth, label=f'Q1: {q1:,.2f}')
        plt.axhline(limite_inferior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Inf: {limite_inferior:,.2f}')
        
    else:
        plt.axhline(limite_superior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Sup: {limite_superior:,.2f}')
        plt.axhline(q3, color='green', linestyle='--', linewidth=linewidth, label=f'Q3: {q3:,.2f}')
        plt.axhline(media, color='purple', linestyle=':', linewidth=linewidth+0.5, label=f'Média: {media:,.2f}')
        plt.axhline(mediana, color='orange', linestyle='-', linewidth=linewidth+0.5, label=f'Mediana: {mediana:,.2f}')
        plt.axhline(q1, color='red', linestyle='--', linewidth=linewidth, label=f'Q1: {q1:,.2f}')
        plt.axhline(limite_inferior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Inf: {limite_inferior:,.2f}')
        

    plt.title(f"{titulo}\n(IQR: {iqr:.2f}, Limites: {fator}×IQR)", fontsize=12)
    
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.
    )
    plt.show()"""
    
def plotar_boxplot(df):
    """Gera boxplot com opções de transformação logarítmica, zoom e limites personalizados"""
    print("\n" + "="*50)
    print("📊 CONFIGURAÇÃO DO BOXPLOT")
    
    # 1. Seleção da coluna
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print("\n🔢 COLUNAS NUMÉRICAS DISPONÍVEIS:")
    for i, col in enumerate(colunas_numericas, 1):
        print(f"{i}. {col}")
    
    try:
        col_idx = int(input("\n▶ Escolha o número da coluna: ")) - 1
        coluna = colunas_numericas[col_idx]
    except (ValueError, IndexError):
        print("❌ Seleção inválida!")
        return

    # Mostrar estatísticas básicas para referência
    stats = df[coluna].describe()
    print(f"\nℹ️ Estatísticas de '{coluna}':")
    print(f"• Mínimo: {stats['min']:,.2f}")
    print(f"• Máximo: {stats['max']:,.2f}")
    print(f"• Média: {stats['mean']:,.2f}")
    print(f"• Mediana: {stats['50%']:,.2f}")

    # 2. Configurações básicas
    print("\n⚙️ CONFIGURAÇÕES BÁSICAS:")
    fator = float(input("▶ Fator do IQR (1.5 ou 3.0): ") or "1.5")
    orientacao = input("▶ Orientação (h/v): ").lower() or "v"
    
    # Configuração de limites personalizados
    print("\n🔘 LIMITES PERSONALIZADOS (deixe em branco para usar valores calculados)")
    try:
        min_personalizado = input(f"▶ Valor mínimo (sugerido: {stats['min']:.2f}): ")
        min_personalizado = float(min_personalizado) if min_personalizado else None
        
        max_personalizado = input(f"▶ Valor máximo (sugerido: {stats['max']:.2f}): ")
        max_personalizado = float(max_personalizado) if max_personalizado else None
    except ValueError:
        print("❌ Valor inválido! Usando limites calculados automaticamente.")
        min_personalizado, max_personalizado = None, None

    # Tamanho da figura
    while True:
        try:
            largura = float(input("▶ Largura da figura (1-20, padrão=10): ") or 10)
            altura = float(input("▶ Altura da figura (1-20, padrão=6): ") or 6)
            if 1 <= largura <= 20 and 1 <= altura <= 20:
                break
            print("⚠️ Valores devem estar entre 1 e 20")
        except ValueError:
            print("⚠️ Digite números válidos")
            largura, altura = 10, 6

    # 3. Tipo de visualização
    print("\n🔍 MELHORIA DE VISUALIZAÇÃO:")
    print("1. Padrão (recomendado para dados uniformes)")
    print("2. Escala logarítmica (recomendado para outliers extremos)")
    print("3. Zoom na área interquartil (foco no núcleo dos dados)")
    print("4. Retirar outliers da visualização")
    escolha_visualizacao = input("▶ Escolha (1/2/3/4): ") or "1"

    # Cálculos estatísticos
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1
    mediana = df[coluna].median()
    media = df[coluna].mean()
    
    # Usa limites personalizados ou calculados
    limite_inferior = min_personalizado if min_personalizado is not None else max(0, q1 - fator * iqr)
    limite_superior = max_personalizado if max_personalizado is not None else q3 + fator * iqr

    # Plotagem
    plt.figure(figsize=(largura, altura), tight_layout=True)
    sns.set_style("whitegrid")
    
    # Filtra os dados dentro dos limites
    dados_filtrados = df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)][coluna]
    
    showfliers = escolha_visualizacao != "4"
    if orientacao == "h":
        ax = sns.boxplot(y=dados_filtrados, whis=fator, color="skyblue", showfliers=showfliers)
    else:
        ax = sns.boxplot(x=dados_filtrados, whis=fator, color="skyblue", showfliers=showfliers)

    # Aplica a melhoria escolhida
    if escolha_visualizacao == "2":
        if orientacao == "h":
            plt.yscale('log')
            plt.ylabel(f"{coluna} (escala logarítmica)")
        else:
            plt.xscale('log')
            plt.xlabel(f"{coluna} (escala logarítmica)")
        titulo = f"Boxplot LOGARÍTMICO de '{coluna}'"
    elif escolha_visualizacao == "3":
        limite_superior_zoom = q3 + 5 * iqr
        if orientacao == "h":
            plt.ylim(0, limite_superior_zoom)
        else:
            plt.xlim(0, limite_superior_zoom)
        titulo = f"Boxplot de '{coluna}' (Zoom IQR)"
    else:
        titulo = f"Boxplot de '{coluna}'"

    # Linhas de referência
    linewidth = 2.5 if escolha_visualizacao != "3" else 3.0
    if orientacao == "h":
        plt.axhline(limite_superior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Sup: {limite_superior:,.2f}')
        plt.axhline(q3, color='green', linestyle='--', linewidth=linewidth, label=f'Q3: {q3:,.2f}')
        plt.axhline(media, color='purple', linestyle=':', linewidth=linewidth+0.5, label=f'Média: {media:,.2f}')
        plt.axhline(mediana, color='orange', linestyle='-', linewidth=linewidth+0.5, label=f'Mediana: {mediana:,.2f}')
        plt.axhline(q1, color='red', linestyle='--', linewidth=linewidth, label=f'Q1: {q1:,.2f}')
        plt.axhline(limite_inferior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Inf: {limite_inferior:,.2f}')
    else:
        plt.axvline(limite_superior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Sup: {limite_superior:,.2f}')
        plt.axvline(q3, color='green', linestyle='--', linewidth=linewidth, label=f'Q3: {q3:,.2f}')
        plt.axvline(media, color='purple', linestyle=':', linewidth=linewidth+0.5, label=f'Média: {media:,.2f}')
        plt.axvline(mediana, color='orange', linestyle='-', linewidth=linewidth+0.5, label=f'Mediana: {mediana:,.2f}')
        plt.axvline(q1, color='red', linestyle='--', linewidth=linewidth, label=f'Q1: {q1:,.2f}')
        plt.axvline(limite_inferior, color='gray', linestyle='-.', linewidth=1, label=f'Lim Inf: {limite_inferior:,.2f}')

    # Adiciona informação sobre os limites usados
    if min_personalizado is not None or max_personalizado is not None:
        titulo += "\n(Limites personalizados)"
    
    plt.title(f"{titulo}\n(IQR: {iqr:.2f}, Limites: {fator}×IQR)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    

def tratamento_limpeza(df_original): 
    df = df_original.copy()
    log_operacoes = []
    
    while True:
        print("\n" + "="*50)
        print("🧹 MENU DE TRATAMENTO/LIMPEZA")
        print("="*50)
        print("1. Excluir linhas")
        print("2. Alterar tipo de dados")
        print("3. Excluir colunas")
        print("4. Tratar valores nulos")
        print("5. Salvar dataset modificado")
        print("0. Voltar ao menu principal (descartar alterações)")
        
        opcao = input("\n▶ Escolha uma opção: ")
        
        if opcao == '0':
            return None  # Descarta alterações
        elif opcao == '1':
            df, log = excluir_linhas(df)
            log_operacoes.extend(log)
        elif opcao == '2':
            df, log = alterar_tipo(df)
            log_operacoes.extend(log)
        elif opcao == '3':
            df, log = excluir_colunas(df)
            log_operacoes.extend(log)
        elif opcao == '4':
            df, log = tratar_nulos(df)
            log_operacoes.extend(log)
        elif opcao == '5':
            salvar_dataset(df, log_operacoes)
            return df
        else:
            print("❌ Opção inválida!")

def excluir_linhas(df):
    log = []
    
    # Mostra colunas numéricas para seleção
    numericas = df.select_dtypes(include=['int64', 'float64']).columns
    print("\n🔢 COLUNAS NUMÉRICAS DISPONÍVEIS:")
    for i, col in enumerate(numericas, 1):
        print(f"{i}. {col} (Tipo: {df[col].dtype})")
    
    # Validação da seleção da coluna
    while True:
        try:
            col_idx = int(input("\n▶ Selecione a coluna para filtrar: ")) - 1
            if 0 <= col_idx < len(numericas):
                coluna = numericas[col_idx]
                break
            print("❌ Número inválido. Tente novamente.")
        except ValueError:
            print("❌ Por favor, digite apenas números.")

    print("\n⚙️ CRITÉRIOS DE EXCLUSÃO:")
    print("1. Valores nulos/NaN")
    print("2. Valores acima de X")
    print("3. Valores abaixo de X")
    print("4. Valores fora do intervalo IQR")
    
    # Validação do critério
    while True:
        criterio = input("▶ Escolha o critério: ")
        if criterio in ['1', '2', '3', '4']:
            break
        print("❌ Opção inválida. Digite 1, 2, 3 ou 4")

    linhas_removidas = 0
    
    if criterio == '1':
        linhas_removidas = df[coluna].isna().sum()
        if linhas_removidas > 0:
            df = df.dropna(subset=[coluna])
            log.append(f"Removidas {linhas_removidas} linhas com valores nulos na coluna '{coluna}'")
        else:
            print("⚠️ Nenhum valor nulo encontrado nesta coluna.")
    
    elif criterio == '2':
        while True:
            try:
                limite = float(input("▶ Digite o valor limite superior: "))
                linhas_removidas = len(df[df[coluna] > limite])
                if linhas_removidas > 0:
                    df = df[df[coluna] <= limite]
                    log.append(f"Removidas {linhas_removidas} linhas com valores acima de {limite} na coluna '{coluna}'")
                    break
                print("⚠️ Nenhum valor acima do limite encontrado. Tente um valor maior.")
            except ValueError:
                print("❌ Por favor, digite um número válido.")
    
    elif criterio == '3':
        while True:
            try:
                limite = float(input("▶ Digite o valor limite inferior: "))
                linhas_removidas = len(df[df[coluna] < limite])
                if linhas_removidas > 0:
                    df = df[df[coluna] >= limite]
                    log.append(f"Removidas {linhas_removidas} linhas com valores abaixo de {limite} na coluna '{coluna}'")
                    break
                print("⚠️ Nenhum valor abaixo do limite encontrado. Tente um valor menor.")
            except ValueError:
                print("❌ Por favor, digite um número válido.")
    
    elif criterio == '4':
        q1 = df[coluna].quantile(0.25)
        q3 = df[coluna].quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5*iqr
        limite_sup = q3 + 1.5*iqr
        linhas_removidas = len(df[(df[coluna] < limite_inf) | (df[coluna] > limite_sup)])
        if linhas_removidas > 0:
            df = df[(df[coluna] >= limite_inf) & (df[coluna] <= limite_sup)]
            log.append(f"Removidas {linhas_removidas} linhas outliers (IQR) na coluna '{coluna}'")
        else:
            print("⚠️ Nenhum outlier encontrado usando o método IQR")

    if linhas_removidas > 0:
        print(f"\n✅ Total de linhas removidas: {linhas_removidas}")
        print(f"📊 Dataset resultante: {len(df)} linhas")
    else:
        print("\nℹ️ Nenhuma linha foi removida")
    
    return df, log

def salvar_dataset(df, log_operacoes):
    print("\n" + "="*50)
    print("💾 SALVAR DATASET MODIFICADO")
    
    sufixo = input("▶ Digite o sufixo para o novo arquivo (ex: 'clean' para 'vehicles_clean.feather'): ")
    formato = input("▶ Formato de saída (1-feather, 2-csv): ") or "1"
    
    nome_base = ARQUIVO_FEATHER.split('.')[0]
    novo_nome = f"{nome_base}_{sufixo}.{'feather' if formato == '1' else 'csv'}"
    
    if formato == '1':
        df.to_feather(novo_nome)
    else:
        df.to_csv(novo_nome, index=False)
    
    print("\n📝 LOG DE OPERAÇÕES REALIZADAS:")
    for operacao in log_operacoes:
        print(f"• {operacao}")
    
    print(f"\n✅ Dataset salvo como: {novo_nome}")

def alterar_tipo(df):
    log = []
    print("\n📋 COLUNAS DISPONÍVEIS:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col} (Tipo atual: {df[col].dtype})")
    
    col_idx = int(input("\n▶ Selecione a coluna para conversão: ")) - 1
    coluna = df.columns[col_idx]
    
    print("\n📝 TIPOS DISPONÍVEIS:")
    print("1. Inteiro (int64)")
    print("2. Decimal (float64)")
    print("3. Texto (object)")
    print("4. Categórico (category)")
    novo_tipo = input("▶ Escolha o novo tipo: ")
    
    try:
        if novo_tipo == '1':
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce').astype('int64')
        elif novo_tipo == '2':
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce').astype('float64')
        elif novo_tipo == '3':
            df[coluna] = df[coluna].astype('object')
        elif novo_tipo == '4':
            df[coluna] = df[coluna].astype('category')
        
        log.append(f"Coluna '{coluna}' convertida para {df[coluna].dtype}")
        print(f"✅ Tipo alterado com sucesso para {df[coluna].dtype}")
    except Exception as e:
        print(f"❌ Erro na conversão: {e}")
    
    return df, log

def tratar_nulos(df):
    log = []
    print("\n🧹 TRATAMENTO DE VALORES NULOS")
    
    colunas_com_nulos = df.columns[df.isnull().any()].tolist()
    if not colunas_com_nulos:
        print("✅ Nenhuma coluna com valores nulos encontrada!")
        return df, log
    
    print("\n📋 COLUNAS COM VALORES NULOS:")
    for i, col in enumerate(colunas_com_nulos, 1):
        nulos = df[col].isnull().sum()
        print(f"{i}. {col} ({nulos} nulos, {nulos/len(df):.1%})")
    
    col_idx = int(input("\n▶ Selecione a coluna para tratamento: ")) - 1
    coluna = colunas_com_nulos[col_idx]
    
    print("\n⚙️ MÉTODOS DE TRATAMENTO:")
    print("1. Remover linhas com nulos")
    print("2. Preencher com média (numéricas)")
    print("3. Preencher com mediana (numéricas)")
    print("4. Preencher com moda (categóricas)")
    metodo = input("▶ Escolha o método: ")
    
    nulos_antes = df[coluna].isnull().sum()
    
    if metodo == '1':
        df = df.dropna(subset=[coluna])
        log.append(f"Removidas {nulos_antes} linhas com nulos na coluna '{coluna}'")
    elif metodo == '2' and pd.api.types.is_numeric_dtype(df[coluna]):
        fill_value = df[coluna].mean()
        df[coluna] = df[coluna].fillna(fill_value)
        log.append(f"Preenchidos {nulos_antes} nulos com média ({fill_value:.2f}) na coluna '{coluna}'")
    # [...] outros métodos
    
    print(f"\n✅ {nulos_antes} valores nulos tratados na coluna '{coluna}'")
    return df, log

def plotar_dispersao(df):
    """Gera gráfico de dispersão entre duas variáveis com personalização"""
    print("\n" + "="*50)
    print("📊 GRÁFICO DE DISPERSÃO")
    
    # Seleção das colunas numéricas
    numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numericas) < 2:
        print("❌ É necessário ter pelo menos 2 colunas numéricas!")
        return
    
    print("\n🔢 COLUNAS NUMÉRICAS DISPONÍVEIS:")
    for i, col in enumerate(numericas, 1):
        print(f"{i}. {col}")
    
    # Seleção das colunas
    try:
        print("\nSelecione as colunas para o eixo X e Y:")
        x_idx = int(input("▶ Número da coluna para eixo X: ")) - 1
        y_idx = int(input("▶ Número da coluna para eixo Y: ")) - 1
        
        if x_idx == y_idx:
            print("❌ As colunas devem ser diferentes!")
            return
            
        if not (0 <= x_idx < len(numericas) and 0 <= y_idx < len(numericas)):
            print("❌ Números inválidos!")
            return
            
        x_col = numericas[x_idx]
        y_col = numericas[y_idx]
    except ValueError:
        print("❌ Digite apenas números!")
        return
    
    # Configurações do gráfico
    print("\n⚙️ CONFIGURAÇÕES DO GRÁFICO:")
    
    # Tamanho da figura
    print("\n● Tamanho da figura (5-20):")
    print("   - Padrão: 10x6 | Use valores maiores para muitos dados")
    while True:
        try:
            largura = float(input("▶ Largura (5-20, padrão=10): ") or 10)
            altura = float(input("▶ Altura (5-20, padrão=6): ") or 6)
            if 5 <= largura <= 20 and 5 <= altura <= 20:
                break
            print("⚠️ Valores devem estar entre 5 e 20")
        except ValueError:
            print("⚠️ Digite números válidos")
    
    # Configurações de pontos
    print("\n● Configuração dos pontos:")
    print("   - Tamanho (1-100):\n     Padrão=20 | Use 5-15 para muitos pontos, 30+ para poucos")
    print("   - Opacidade (0.1-1.0):\n     Padrão=0.7 | Use valores baixos (0.2-0.5) para datasets grandes")
    print("   - Cores disponíveis: blue, red, green, purple, orange, etc.")
    
    cor = input("▶ Cor dos pontos (padrão=blue): ") or "blue"
    tamanho = float(input("▶ Tamanho (1-100, padrão=20): ") or 20)
    opacidade = float(input("▶ Opacidade (0.1-1.0, padrão=0.7): ") or 0.7)
    
    # Escala logarítmica
    print("\n● Escala logarítmica:")
    print("   - RECOMENDÁVEL PARA:")
    print("     • Dados assimétricos (média ≠ mediana)")
    print("     • Valores com grande amplitude (ex: 1 a 1.000.000)")
    print("     • Quando há muitos outliers extremos")
    print("   - NÃO RECOMENDADO PARA:")
    print("     • Dados negativos ou com valores zero")
    print("     • Dados já normalizados")
    log_scale = input("▶ Aplicar escala logarítmica? (s/n, padrão=n): ").lower() == 's'

    # Linha de regressão
    print("\n● Linha de regressão:")
    print("   - RECOMENDÁVEL PARA:")
    print("     • Identificar tendências lineares")
    print("     • Verificar direção (positiva/negativa) da relação")
    print("   - EVITE quando:")
    print("     • Os dados são categóricos")
    print("     • Há relação não-linear clara")
    regressao = input("▶ Adicionar linha de regressão? (s/n, padrão=n): ").lower() == 's'
    
    # Títulos
    titulo = input(f"▶ Título do gráfico (deixe em branco para padrão): ") or f"Dispersão: {x_col} vs {y_col}"
    x_label = input(f"▶ Rótulo do eixo X (deixe em branco para '{x_col}'): ") or x_col
    y_label = input(f"▶ Rótulo do eixo Y (deixe em branco para '{y_col}'): ") or y_col
    
    # Criar o gráfico
    plt.figure(figsize=(largura, altura))
    sns.set_style("whitegrid")
    
    if regressao:
        sns.regplot(x=x_col, y=y_col, data=df, 
                   scatter_kws={'color': cor, 's': tamanho, 'alpha': opacidade},
                   line_kws={'color': 'red', 'linestyle': '--'})
    else:
        sns.scatterplot(x=x_col, y=y_col, data=df, 
                       color=cor, s=tamanho, alpha=opacidade)
    
    # Aplicar escala log se selecionado
    if log_scale:
        if pd.api.types.is_numeric_dtype(df[x_col]):
            plt.xscale('log')
            x_label = f"{x_label} (escala log)"
        if pd.api.types.is_numeric_dtype(df[y_col]):
            plt.yscale('log')
            y_label = f"{y_label} (escala log)"
    
    # Configurações adicionais
    plt.title(titulo, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Adicionar grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mostrar correlação se for numérico
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        corr = df[[x_col, y_col]].corr().iloc[0,1]
        plt.text(0.95, 0.95, f"Correlação: {corr:.2f}", 
                transform=plt.gca().transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Ajustar layout
    plt.tight_layout()
    plt.show()

def plotar_histograma(df):
    """Gera histograma com configurações totalmente manuais"""
    print("\n" + "="*50)
    print("📊 HISTOGRAMA DE CONTAGEM (CONFIGURAÇÃO MANUAL)")
    
    # Seleção da coluna
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not colunas_numericas:
        print("❌ Nenhuma coluna numérica encontrada!")
        return
    
    print("\n🔢 COLUNAS NUMÉRICAS DISPONÍVEIS:")
    for i, col in enumerate(colunas_numericas, 1):
        print(f"{i}. {col}")
    
    try:
        col_idx = int(input("\n▶ Selecione o número da coluna: ")) - 1
        coluna = colunas_numericas[col_idx]
    except (ValueError, IndexError):
        print("❌ Seleção inválida!")
        return

    # Mostrar estatísticas rápidas para referência
    stats = df[coluna].describe()
    print(f"\nℹ️ Estatísticas rápidas de '{coluna}':")
    print(f"• Mínimo: {stats['min']:,.2f}")
    print(f"• Máximo: {stats['max']:,.2f}")
    print(f"• Média: {stats['mean']:,.2f}")
    print(f"• Mediana: {stats['50%']:,.2f}")

    # Configurações básicas
    print("\n⚙️ CONFIGURAÇÕES BÁSICAS:")
    bins = int(input("▶ Número de bins: ") or 10)
    cor = input("▶ Cor (blue/green/red/purple/orange/gray): ") or "blue"
    largura = float(input("▶ Largura da figura (5-20): ") or 12)
    altura = float(input("▶ Altura da figura (5-20): ") or 8)

    # Configurações de faixa de valores
    print("\n🔢 LIMITES DO EIXO X (Deixe em branco para automático):")
    min_x = input(f"▶ Valor mínimo (atual {stats['min']:,.2f}): ")
    max_x = input(f"▶ Valor máximo (atual {stats['max']:,.2f}): ")
    
    try:
        min_x = float(min_x) if min_x else None
        max_x = float(max_x) if max_x else None
    except ValueError:
        print("⚠️ Valores inválidos. Usando automático.")
        min_x, max_x = None, None

    # Configurações avançadas
    print("\n🔧 CONFIGURAÇÕES AVANÇADAS:")
    print("Escalas disponíveis:")
    print("1. Linear (padrão)")
    print("2. Log no eixo Y")
    print("3. Log no eixo X")
    print("4. Log em ambos os eixos")
    escala = input("▶ Escolha a escala (1-4): ") or "1"
    
    densidade = input("▶ Mostrar curva de densidade? (s/n): ").lower() == 's'
    acumulado = input("▶ Histograma acumulado? (s/n): ").lower() == 's'
    mostrar_stats = input("▶ Mostrar estatísticas? (s/n): ").lower() == 's'

    # Plotagem
    plt.figure(figsize=(largura, altura))
    sns.set_style("whitegrid")
    
    ax = sns.histplot(
        data=df, 
        x=coluna, 
        bins=bins, 
        color=cor,
        kde=densidade,
        cumulative=acumulado,
        stat='density' if densidade else 'count',
        log_scale=(True if escala in ['3','4'] else False, 
                  True if escala in ['2','4'] else False)
    )
    
    # Aplicar limites do eixo X se especificados
    if min_x is not None or max_x is not None:
        ax.set_xlim(left=min_x, right=max_x)
    
    # Título e labels
    titulo = f"Histograma de {coluna}"
    if densidade:
        titulo += " com Densidade"
    if acumulado:
        titulo += " Acumulado"
    
    plt.title(titulo)
    plt.xlabel(coluna + (" (log)" if escala in ['3','4'] else ""))
    plt.ylabel(("Densidade" if densidade else "Contagem") + (" (log)" if escala in ['2','4'] else ""))

    # Estatísticas se solicitado
    if mostrar_stats:
        stats_text = (f"Mínimo: {stats['min']:,.2f}\n"
                     f"Máximo: {stats['max']:,.2f}\n"
                     f"Média: {stats['mean']:,.2f}\n"
                     f"Mediana: {stats['50%']:,.2f}\n"
                     f"Std: {stats['std']:,.2f}")
        
        plt.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    # Opção de salvar
    if input("\n💾 Salvar gráfico? (s/n): ").lower() == 's':
        nome = input("▶ Nome do arquivo (sem extensão): ") or f"hist_{coluna}"
        plt.savefig(f"{nome}.png", dpi=300)
        print(f"✅ Salvo como {nome}.png")
    
    plt.show()

def valores_distintos(df):
    """Mostra valores distintos e contagens de uma coluna"""
    print("\n" + "="*50)
    print("📌 COLUNAS DISPONÍVEIS:")
    
    for i, coluna in enumerate(df.columns, 1):
        print(f"{i}. {coluna}")
    
    try:
        col_idx = int(input("\n▶ Selecione o número da coluna: ")) - 1
        coluna = df.columns[col_idx]
        
        print(f"\n📊 Valores distintos na coluna '{coluna}':")
        print(df[coluna].value_counts(dropna=False).to_string())
        
    except (ValueError, IndexError):
        print("❌ Seleção inválida!")

def agrupar_por_faixas(df):
    """Agrupa registros por faixas de valores de uma coluna numérica"""
    print("\n" + "="*50)
    print("📊 AGRUPAMENTO POR FAIXAS DE VALORES")
    
    # Selecionar apenas colunas numéricas
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not colunas_numericas:
        print("❌ Nenhuma coluna numérica encontrada no dataset!")
        return
    
    print("\n🔢 COLUNAS NUMÉRICAS DISPONÍVEIS:")
    for i, col in enumerate(colunas_numericas, 1):
        print(f"{i}. {col}")
    
    try:
        col_idx = int(input("\n▶ Selecione o número da coluna: ")) - 1
        coluna = colunas_numericas[col_idx]
    except (ValueError, IndexError):
        print("❌ Seleção inválida!")
        return
    
    # Mostrar estatísticas básicas para referência
    stats = df[coluna].describe()
    print(f"\nℹ️ Estatísticas de '{coluna}':")
    print(f"• Mínimo: {stats['min']:,.0f}")
    print(f"• Máximo: {stats['max']:,.0f}")
    print(f"• Média: {stats['mean']:,.0f}")
    print(f"• Mediana: {stats['50%']:,.0f}")
    
    # Obter configurações do usuário
    print("\n⚙️ CONFIGURAÇÃO DAS FAIXAS:")
    
    # Validação do valor mínimo
    while True:
        try:
            minimo = float(input(f"▶ Valor mínimo (sugerido: {stats['min']:.2f}): ") or stats['min'])
            break
        except ValueError:
            print("❌ Por favor, digite um número válido.")
    
    # Validação do valor máximo
    while True:
        try:
            maximo = float(input(f"▶ Valor máximo (sugerido: {stats['max']:.2f}): ") or stats['max'])
            if maximo > minimo:
                break
            print(f"❌ O valor máximo deve ser maior que o mínimo ({minimo})")
        except ValueError:
            print("❌ Por favor, digite um número válido.")
    
    # Validação do intervalo
    while True:
        try:
            intervalo = float(input("▶ Intervalo de cada faixa (ex: 50 para faixas de 50 em 50): "))
            if intervalo > 0:
                break
            print("❌ O intervalo deve ser maior que zero")
        except ValueError:
            print("❌ Por favor, digite um número válido.")
    
    # Perguntar sobre a quantidade de linhas a serem exibidas ANTES de processar
    try:
        max_rows = int(input("\n▶ Quantidade de linhas a exibir (0 para todas): ") or 0)
    except ValueError:
        print("❌ Valor inválido. Mostrando todas as linhas.")
        max_rows = 0
    
    # Criar as faixas
    faixas = []
    inicio = minimo
    
    while inicio < maximo:
        fim = inicio + intervalo
        # Garante que a última faixa não ultrapasse o máximo
        if fim > maximo:
            fim = maximo
        faixas.append((inicio, fim))
        inicio = fim
    
    # Contar registros em cada faixa
    resultados = []
    for i, (inicio_faixa, fim_faixa) in enumerate(faixas):
        if i == len(faixas) - 1:
            # Última faixa inclui o valor máximo
            condicao = (df[coluna] >= inicio_faixa) & (df[coluna] <= fim_faixa)
        else:
            condicao = (df[coluna] >= inicio_faixa) & (df[coluna] < fim_faixa)
        
        contagem = len(df[condicao])
        porcentagem = (contagem / len(df)) * 100
        
        resultados.append({
            'Faixa': f"{inicio_faixa:,.0f} a {fim_faixa:,.0f}",
            'Contagem': contagem,
            'Porcentagem (%)': porcentagem
        })
    
    # Criar DataFrame com os resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Configurar formatação para melhor visualização
    pd.options.display.float_format = '{:,.2f}'.format
    
    # Aplicar a restrição de linhas
    if max_rows > 0:
        if len(df_resultados) > max_rows:
            print(f"\nℹ️ Mostrando {max_rows} linhas. {len(df_resultados)-max_rows} linhas ocultas.")
            df_resultados = df_resultados.head(max_rows)
        pd.options.display.max_rows = None  # Desabilita o truncamento do pandas
    else:
        pd.options.display.max_rows = None
    
    print("\n" + "="*50)
    print(f"📊 RESULTADO - DISTRIBUIÇÃO POR FAIXAS DE '{coluna}'")
    print("="*50)
    print(df_resultados.to_string(index=False))
    print("="*50)
    
    # Resetar configurações de exibição
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_rows')
    
    # Opção para plotar gráfico
    if input("\n▶ Deseja visualizar um gráfico? (s/n): ").lower() == 's':
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Faixa', y='Contagem', data=df_resultados, palette='viridis')
        plt.title(f"Distribuição de registros por faixas de {coluna}")
        plt.xlabel("Faixas de valores")
        plt.ylabel("Número de registros")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# =============================================== #
# ================== MENU ====================== #
def mostrar_menu():
    print("\n" + "="*50)
    print("📊 MENU PRINCIPAL DE ANÁLISE")
    print("="*50)
    print("1. Estatísticas das colunas numéricas")
    print("2. Comparação entre dois campos")
    print("3. Boxplot com limites personalizados")
    print("4. Tratamento/Limpeza de dados")
    print("5. Valores distintos de uma coluna")
    print("6. Gráfico de dispersão")
    print("7. Histograma de contagem")
    print("8. Agrupamento por faixas de valores")  # NOVA OPÇÃO
    print("0. Sair")
    return input("▶ Escolha uma opção: ")

# =============================================== #
# ================== MAIN ======================= #
if __name__ == "__main__":
    while True:
        opcao = mostrar_menu()
        
        if opcao == '0':
            print("\n✅ Programa encerrado. Até logo!")
            break
        elif opcao == '1':
            estatisticas_colunas(df)
        elif opcao == '2':
            comparacao_campos(df)
        elif opcao == '3':
            plotar_boxplot(df)
        elif opcao == '4':
            resultado = tratamento_limpeza(df)
            if resultado is not None:
                df = resultado
        elif opcao == '5':
            valores_distintos(df)
        elif opcao == '6':  # NOVA OPÇÃO
            plotar_dispersao(df)
        elif opcao == '7':
            plotar_histograma(df)
        elif opcao == '8':
            agrupar_por_faixas(df)
        else:
            print("❌ Opção inválida!")