"""Portuguese (pt-BR) strings for the animated figures.

Every piece of text that ends up rendered inside a GIF lives here. Keeping it
in a single module makes the language boundary explicit: the rest of the code
base, including the static PNG figures, the console output and the report, is
in English.
"""

from __future__ import annotations

CREDIT = "Base de faces Olivetti (AT&T/ORL) · 400 imagens · 40 indivíduos"

RULES = {
    "hebb": "Hebb (Kosko, 1988)",
    "multiple_training": "Múltiplos treinos",
    "least_squares": "Mínimos quadrados",
    "dense": "Densa (exponencial)",
}

CODEBOOKS = {
    "hadamard": "Códigos de Hadamard",
    "random": "Códigos aleatórios",
    "correlated": "Códigos correlacionados",
}

# --- Panels -----------------------------------------------------------------
PANEL_X_STATE = "Camada X — a face"
PANEL_X_ERROR = "Bits errados"
PANEL_Y_STATE = "Camada Y — o código"
PANEL_ENERGY = "Energia de Lyapunov"
PANEL_PROBE = "Sonda apresentada"
PANEL_RECOVERED = "Face recuperada"
PANEL_TEST_PHOTO = "Fotografia inédita"
PANEL_ANSWER = "Quem a rede diz que é"
PANEL_PROTOTYPE = "Protótipo evocado do código"
PANEL_NETWORK_CODE = "Código produzido pela rede (X→Y)"

# --- Axes -------------------------------------------------------------------
AXIS_HALF_ITER = "Meia-iteração (uma atualização de camada)"
AXIS_ENERGY = "Energia  E(x, y)"
AXIS_STORED_PAIRS = "Pares armazenados  P"
AXIS_PERFECT_RECALL = "Faces recuperadas sem erro"
AXIS_CORRUPTION = "Fração da face corrompida"

# --- Dynamics ---------------------------------------------------------------
STAGE_CLEAN = "a face armazenada"
STAGE_PROBE_SHOWN = "a sonda entra na rede"
STAGE_FORWARD = "a rede propaga  X → Y"
STAGE_BACKWARD = "a rede propaga  Y → X"
STAGE_ERASE = "apagamos a face — resta apenas o código"
STAGE_STABLE = "par estável"

STATUS_RECALLED = "recuperação exata"
STATUS_SPURIOUS = "estado espúrio"

WRONG_BITS = "bits errados: {value}"
WRONG_SHORT = "{value} de erro"
EXACT_SHORT = "sem erro"
ENERGY_VALUE = "energia: {value}"
STORED_PAIRS_COUNT = "P = {n} pares armazenados"
CORRUPTION_LEVEL = "{value} da face corrompida"
PHOTO_COUNTER = "Fotografia {k} de {n}"

PHASE_LABELS = ("1 · da face ao código e de volta", "2 · só do código à face")

NOISE_LABEL = "Ruído — pixels invertidos"
OCCLUSION_LABEL = "Oclusão — pixels apagados"

TRUE_SUBJECT = "é o indivíduo {k}"
ANSWERED = "respondeu: indivíduo {k}"
CORRECT = "✓ acertou"
WRONG = "✗ errou"
ACCURACY_NOTE = "acerto em fotografias inéditas: {value}"
RED_SQUARES_NOTE = "quadrados vermelhos = bits fora do código correto"


def corrupting(percent: float) -> str:
    """Caption for the pre-roll frames that build the noisy probe."""
    return f"corrompendo a sonda: {percent:.0f}% dos pixels invertidos"


# --- Titles, subtitles and the closing takeaway of each animation -----------
TITLES = {
    "01_bidirectional_recall": (
        "Uma memória que funciona\nnos dois sentidos",
        "Uma face corrompida entra e a rede devolve a face e o seu código.\n"
        "Depois apagamos a face — e o código sozinho a traz de volta.",
    ),
    "02_capacity": (
        "Quantos pares cabem numa\nmemória associativa?",
        "A mesma rede e a mesma regra de aprendizagem.\n"
        "A única diferença são os códigos escolhidos para a camada Y.",
    ),
    "03_robustness": (
        "Quanto dano a memória\nconsegue sobreviver",
        "Duas formas de destruir a sonda: inverter pixels ou apagá-los.\n"
        "A rede reage de maneiras muito diferentes a cada uma.",
    ),
    "04_identification": (
        "Do rosto ao nome,\ne do nome ao rosto",
        "Fotografias que a rede nunca viu · 40 identidades\n"
        "{n_train} poses guardadas por pessoa",
    ),
}

TAKEAWAYS = {
    "01_bidirectional_recall":
        "Um único conjunto de pesos serve aos dois sentidos, e a energia nunca "
        "sobe: a convergência é garantida.",
    "02_capacity":
        "Códigos ortogonais anulam a interferência: 20× mais pares "
        "armazenados, sem mudar nada na rede.",
    "03_robustness":
        "Pixel apagado não mente. A rede tolera 80% de oclusão, contra 45% de "
        "ruído.",
    "04_identification":
        "O sentido inverso não devolve uma foto guardada: sintetiza o "
        "protótipo da pessoa, que nunca foi mostrado à rede.",
}
