import numpy as np
import subprocess

def P_ld():
    combinacoes = [(-1,-1),(-1,1),(1,-1),(1,1)]
    pontos_ld = np.zeros((16,8), dtype=int)
    k = 0

    for a_0, a_1 in combinacoes:
        for b_0, b_1 in combinacoes:
            pontos_ld[k][0] = a_0
            pontos_ld[k][1] = a_1
            pontos_ld[k][2] = b_0
            pontos_ld[k][3] = b_1
            pontos_ld[k][4] = a_0*b_0
            pontos_ld[k][5] = a_0*b_1
            pontos_ld[k][6] = a_1*b_0
            pontos_ld[k][7] = a_1*b_1
            
            k += 1
    return pontos_ld

def gerar_script_polymake(vertices):
    # Formata cada vértice com prefixo 1 (notação homogênea)
    linhas = []
    for v in vertices:
        linha = "[1," + ",".join(str(x) for x in v) + "]"
        linhas.append(linha)
    pontos = ",\n               ".join(linhas)

    script = f"""
use application 'polytope';

my $p = new Polytope(POINTS=>[
               {pontos}
              ]);

print "Numero de facetas: ", $p->N_FACETS, "\\n";
print "\\nFACETAS:\\n";
print $p->FACETS;

combinatorial_symmetries($p);

print "\n", $p->GROUP->FACETS_ACTION->GENERATORS;
"""
    return script

def executar_polymake(script, arquivo="/tmp/politopo_bell.pl"):
    with open(arquivo, "w") as f:
        f.write(script)

    resultado = subprocess.run(
        ["polymake", "--script", arquivo],
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        print("ERRO:")
        print(resultado.stderr)
        return None

    return resultado.stdout

def parsear_facetas(output):
    """Extrai a matriz de facetas do output do polymake"""
    facetas = []
    dentro = False
    for linha in output.split("\n"):
        if "FACETAS:" in linha:
            dentro = True
            continue
        if dentro:
            if linha.strip() == "" or "EQUAÇÕES" in linha:
                dentro = False
                continue
            # Remove < > que o polymake usa para delimitar linhas
            linha = linha.replace("<", "").replace(">", "").strip()
            if linha:
                facetas.append([int(x) for x in linha.split()])
    return np.array(facetas) if facetas else None

# 1. Gera vértices
vertices = P_ld()
print(f"Vértices gerados: {len(vertices)}")
print(vertices)

# 2. Gera script polymake
script = gerar_script_polymake(vertices)

# 3. Executa polymake
print("\nExecutando polymake...\n")
output = executar_polymake(script)
print(output)

# 4. Parseia e exibe facetas
facetas = parsear_facetas(output)
print(facetas)
labels = ["<A_0>", "<A_1>", "<B_0>", "<B_1>",
          "E_00", "E_01", "E_10", "E_11"]
if facetas is not None:
    print(f"\nMatriz de facetas: shape {facetas.shape}")
    b = facetas[:, 0]
    A = facetas[:, 1:]
    print("\nDesigualdades (A @ x <= b):")
    for i, (ai, bi) in enumerate(zip(A, b)):
        termos = " + ".join(
            f"({aj}){labels[j]}" for j, aj in enumerate(ai) if aj != 0
        )
        print(f"  Faceta {i}: {termos} <= {bi}")








