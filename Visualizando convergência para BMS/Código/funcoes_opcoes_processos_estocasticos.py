import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

# Montando uma função que calcula uma opção de call com BMS
def call_options_BMS(S0: float, X:float, T:float, vol:float, r: float) -> float:
    d1 = (np.log(S0/X) + (r + 0.5 * vol**2) * T)/(vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    c = S0 * sps.norm.cdf(d1) - X * np.exp(-r * T) * sps.norm.cdf(d2) 
    return c

# Montando uma função que calcula uma opção de put com BMS
def put_options_BMS(S0: float, X:float, T:float, vol:float, r: float) -> float:
    d1 = (np.log(S0/X) + (r + 0.5 * vol**2) * T)/(vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    p = - S0 * sps.norm.cdf(-d1) + X * np.exp(-r * T) * sps.norm.cdf(-d2)
    return p

# Montando uma função que calcula uma opção de call via método de  monte carlo
def call_options_monte_carlo(S0: float, X:float, T:float, vol:float, r: float, N=1000000) -> float:
    # Simulando os N valores aleatórios normalmente distribuidos
    z = np.random.standard_normal(int(N))
    
    # Calculando o preço do ativo atraves de um movimento browniano geométrico
    ST = S0 * np.exp((r - 0.5 * vol ** 2) * T + vol * np.sqrt(T) * z)
    
    # Tomando os payoffs de cada cenário simulado
    pT = np.maximum(ST - X, 0)
    
    # Tomando a média dos payoffs e multiplicando pelo fator de desconto contínuo e^-rT
    c = np.mean(pT) * np.exp(-r * T)
    return c

# Montando uma função que calcula uma opção de put via método de monte carlo
def put_options_monte_carlo(S0: float, X:float, T:float, vol:float, r: float, N=1000000) -> float:
    # Simulando os N valores aleatórios normalmente distribuidos
    z = np.random.standard_normal(int(N))
    
    # Calculando o preço do ativo atraves de um movimento browniano geométrico
    ST = S0 * np.exp((r - 0.5 * vol ** 2) * T + vol * np.sqrt(T) * z)
    
    # Tomando os payoffs de cada cenário simulado
    pT = np.maximum(X - ST, 0)
    
    # Tomando a média dos payoffs e multiplicando pelo fator de desconto contínuo e^-rT
    p = np.mean(pT) * np.exp(-r * T)
    return p

# Montando uma função que calcula uma opção de call com modelo binomial
def call_options_binomial_tree_model(S0: float, X: float, T:float, vol:float, r:float, N=1000) -> float:
    # Definindo parâmetros utilizados no modelo binomial
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1/u
    growth_rate = np.exp(r * dt)
    p = (growth_rate - d) / (u - d)
    q = 1 - p
    
    # Criando a árvore com todos os valores zerados
    N = int(N)
    tree = [[0.0 for j in range(i+1)] for i in range(N + 1)]
    
    # Calculando o valor das folhas da árvore binomial
    for j in range(N + 1):
        tree[N][j] = np.maximum(S0 * u ** j * d ** (N - j) - X, 0)
        
    # Calcula o os valores dos nós intermediários até a raiz
    for i in reversed(range(N)):
        for j in range(i + 1):
            tree[i][j] = np.exp(-r * dt) * (p * tree[i+1][j+1] + q * tree[i+1][j]) 
          
    # Retorna o valor inicial da árvore de preços
    return tree[0][0]

# Montando uma função que calcula uma opção de put com modelo binomial
def put_options_binomial_tree_model(S0: float, X: float, T:float, vol:float, r:float, N=1000) -> float:
    # Definindo parâmetros utilizados no modelo binomial
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1/u
    growth_rate = np.exp(r * dt)
    p = (growth_rate - d) / (u - d)
    q = 1 - p
    
    # Criando a árvore com todos os valores zerados
    N = int(N)
    tree = [[0.0 for j in range(i+1)] for i in range(N + 1)]
    
    # Calculando o valor das folhas da árvore binomial
    for j in range(N + 1):
        tree[N][j] = np.maximum(X - S0 * u ** j * d ** (N - j), 0)
        
    # Calcula o os valores dos nós intermediários até a raiz
    for i in reversed(range(N)):
        for j in range(i + 1):
            tree[i][j] = np.exp(-r * dt) * (p * tree[i+1][j+1] + q * tree[i+1][j]) 
          
    # Retorna o valor inicial da árvore de preços
    return tree[0][0]

# Montando uma função que visualiza a convergência de monte carlo para BMS em opções de call
def call_monte_carlo_convergence(S0:float, X:float, T:float, vol:float, r:float, lista_N):
    # Obtendo o preço de convergência
    call_price_BMS = call_options_BMS(S, X, T, vol, r)
    
    # Criando uma lista para armazenar o preço de cada opção de call
    call_prices_monte_carlo = []
    
    # Variando o N da simulação de monte carlo para cada N da lista_N
    for N in lista_N:
        c = call_options_monte_carlo(S0, X, T, vol, r, N)
        call_prices_monte_carlo.append(c)
        
    # Elaborando um gráfico para visualizar a convergência
    plt.figure(figsize=(14, 8))
    plt.plot(lista_N, call_prices_monte_carlo, label="Monte carlo", color="blue")
    plt.axhline(y=call_price_BMS, label="BMS", color="black", linestyle="--")
    plt.xlabel("Número de simulações")
    plt.ylabel("Preço")
    plt.title("Observando a convergência do método de monte carlo para BMS em opções de call")
    plt.grid()
    plt.show()
    
# Montando uma função que visualiza a convergência de monte carlo para BMS em opções de put
def put_monte_carlo_convergence(S0:float, X:float, T:float, vol:float, r:float, lista_N):
    # Obteno o preço de convergência
    put_price_BMS = put_options_BMS(S0, X, T, vol, r)
    
    # Criando uma lista para armazenar o preço de cada opção de put
    put_prices_monte_carlo = []
    
    # Variando o N da simulação de monte carlo para cada N da lista_N
    for N in lista_N:
        p = put_options_monte_carlo(S0, X, T, vol, r, N)
        put_prices_monte_carlo.append(p)
        
    # ELaborando um gráfico para visualizar a convergência 
    plt.figure(figsize=(14, 8))
    plt.plot(lista_N, put_prices_monte_carlo, label="Monte carlo", color="red")
    plt.axhline(y=put_price_BMS, label="BMS", color="black", linestyle="--")
    plt.xlabel("Número de simulações")
    plt.ylabel("Preço")
    plt.title("Observando a convergência do método de monte carlo para BMS em opções de put")
    plt.grid()
    plt.show()
    
    
# Montando uma função que visualiza a convergência do modelo binomial para BMS em opções de call
def call_binomial_tree_convergence(S0:float, X:float, T:float, vol:float, r:float, lista_N):
    # Obtendo o preço de convergência
    call_price_BMS = call_options_BMS(S0, X, T, vol, r)
    
    # Criando a lista
    call_prices_binomial_model = []
    
    # Variando o N do modelo binomial para cada N da lista
    for N in lista_N:
        c = call_options_binomial_tree_model(S0, X, T, vol, r, N)
        call_prices_binomial_model.append(c)
        
    # Elaborando um gráfico para visualizar a convergência
    plt.figure(figsize=(14, 8))
    plt.plot(lista_N, call_prices_binomial_model, label="Binomial model", color="blue")
    plt.axhline(y=call_price_BMS, label="BMS", color="black", linestyle="--")
    plt.xlabel("Número de simulações")
    plt.ylabel("Preço")
    plt.title("Observando a convergência do modelo binomial para BMS em opções de call")
    plt.grid()
    plt.show()
    
# Montando uma função que visualiza a convergência do modelo binomial para BMS em opções de put
def put_binomial_tree_convergence(S0:float, X:float, T:float, vol:float, r:float, lista_N):
    # Obtendo o preço de convergência
    put_price_BMS = put_options_BMS(S0, X, T, vol, r)
    
    # Criando a lista
    put_prices_binomial_model = []
    
    # Variando o N do modelo binomial para cada N da lista
    for N in lista_N:
        p = put_options_binomial_tree_model(S0, X, T, vol, r, N)
        put_prices_binomial_model.append(p)
        
    # Elaborando um gráfico para visualizar a convergência
    plt.figure(figsize=(14, 8))
    plt.plot(lista_N, put_prices_binomial_model, label="Binomial model", color="red")
    plt.axhline(y=put_price_BMS, label="BMS", color="black", linestyle="--")
    plt.xlabel("Número de simulações")
    plt.ylabel("Preço")
    plt.title("Observando a convergência do modelo binomial para BMS em opções de put")
    plt.grid()
    plt.show()

# Separando dados de teste gerais e calculando com as funções acima
S = 42
X = 40
r = 0.1
vol = 0.5
T = 0.5

call_option_price_BMS = call_options_BMS(S, X, T, vol, r)
put_option_price_BMS = put_options_BMS(S, X, T, vol, r)
call_option_price_monte_carlo = call_options_monte_carlo(S, X, T, vol, r)
put_option_price_monte_carlo = put_options_monte_carlo(S, X, T, vol, r)
call_option_price_binomial_tree = call_options_binomial_tree_model(S, X, T, vol, r)
put_option_price_binomial_tree = put_options_binomial_tree_model(S, X, T, vol, r)

print(f"Preço da opção de call obtido por BMS: {call_option_price_BMS}")
print(f"Preço da opção de call obtido por monte carlo: {call_option_price_monte_carlo}")
print(f"Preço da opção de call obtido por modelo binomial: {call_option_price_binomial_tree}")
print(f"Preço da opção de put obtido por BMS: {put_option_price_BMS}")
print(f"Preço da opção de put obtido por monte carlo: {put_option_price_monte_carlo}")
print(f"Preço da opção de put obtido por modelo binomial: {put_option_price_binomial_tree}")

# Visualizando as convergências feitas


lista_monte_carlo = np.linspace(0, 50000, 1000)

call_monte_carlo_convergence(S, X, T, vol, r, lista_monte_carlo)
put_monte_carlo_convergence(S, X, T, vol, r, lista_monte_carlo)

lista_binomial_model = np.linspace(1, 250, 100)

call_binomial_tree_convergence(S, X, T, vol, r, lista_binomial_model)
put_binomial_tree_convergence(S, X, T, vol, r, lista_binomial_model)
