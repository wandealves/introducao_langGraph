import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from litellm import completion
import re

class State(TypedDict):
    """ Representa o estado do processo de avaliação da redação """
    redacao: str
    pontuacao_relevancia: float
    pontuacao_gramatica: float
    pontuacao_estrutura: float
    pontuacao_profundidade: float
    pontuacao_final: float

load_dotenv()
llm = 'ollama/gemma2'

def extrair_pontuacao(content: str) -> float:
    """Extrai a pontuação numérica da resposta do LLM."""
    match = re.search(r'Pontuação:\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"Não foi possível extrair a pontuação de: {content}")

def verificar_relevancia(state: State) -> State:
    """Verifica a relevância da redação."""
    print('Verifica a relevância da redação.')
    prompt = """Analise a relevância da seguinte redação em relação ao tema dado, prezando pela excelência da Língua Portuguesa.
                Forneça uma pontuação de relevância entre 0 e 1.
                Sua resposta deve começar com 'Pontuação: ' seguida da pontuação numérica,
                depois forneça sua explicação.\n\nRedação: {redacao}"""
    response=completion(
    model=llm, 
    messages=[{ "content": prompt.format(redacao=state["redacao"]),"role": "user"}], 
    api_base=os.getenv("API_BASE")
    )
    message_content = response.choices[0].message.content
    try:
        state["pontuacao_relevancia"] = extrair_pontuacao(message_content)
    except ValueError as e:
        print(f"Erro em verificar_relevancia: {e}")
        state["pontuacao_relevancia"] = 0.0
    return state

def verificar_gramatica(state: State) -> State:
    """Verifica a gramática da redação."""
    print('Verifica a gramática da redação.')
    prompt = """Analise a gramática da Língua Portuguesa na seguinte redação. 
                Forneça uma pontuação de gramática entre 0 e 1. 
                Sua resposta deve começar com 'Pontuação: ' seguida da pontuação numérica, 
                depois forneça sua explicação.\n\nRedação: {redacao}"""
    response=completion(
    model=llm, 
    messages=[{ "content": prompt.format(redacao=state["redacao"]),"role": "user"}], 
    api_base=os.getenv("API_BASE")
    )
    message_content = response.choices[0].message.content
    try:
        state["pontuacao_gramatica"] = extrair_pontuacao(message_content)
    except ValueError as e:
        print(f"Erro em verificar_gramatica: {e}")
        state["pontuacao_gramatica"] = 0.0
    return state

def analizar_estrutura(state: State) -> State:
    """Analisa a estrutura da redação."""
    print('Analisa a estrutura da redação.')
    prompt = """Analise a estrutura de acordo com a normal culta da Língua Portuguesa na seguinte redação. 
                Forneça uma pontuação de estrutura entre 0 e 1. 
                Sua resposta deve começar com 'Pontuação: ' seguida da pontuação numérica, 
                depois forneça sua explicação.\n\nRedação: {redacao}"""
    response=completion(
    model=llm, 
    messages=[{ "content": prompt.format(redacao=state["redacao"]),"role": "user"}], 
    api_base=os.getenv("API_BASE")
    )
    message_content = response.choices[0].message.content
    try:
        state["pontuacao_estrutura"] = extrair_pontuacao(message_content)
    except ValueError as e:
        print(f"Erro em analizar_estrutura: {e}")
        state["pontuacao_estrutura"] = 0.0
    return state

def avaliar_profundidade(state: State) -> State:
    """Avalia a profundidade de análise na redação."""
    print('Avalia a profundidade de análise na redação.')
    prompt = """Avalie a profundidade de análise na seguinte redação. 
                Forneça uma pontuação de profundidade entre 0 e 1. 
                Sua resposta deve começar com 'Pontuação: ' seguida da pontuação numérica, 
                depois forneça sua explicação.\n\nRedação: {redacao}"""
    response=completion(
    model=llm, 
    messages=[{ "content": prompt.format(redacao=state["redacao"]),"role": "user"}], 
    api_base=os.getenv("API_BASE")
    )
    message_content = response.choices[0].message.content
    try:
        state["pontuacao_profundidade"] = extrair_pontuacao(message_content)
    except ValueError as e:
        print(f"Erro em avaliar_profundidade: {e}")
        state["pontuacao_profundidade"] = 0.0
    return state

def calcular_pontuacao_final(state: State) -> State:
    """Calcula a pontuação final com base nas pontuações dos componentes individuais."""
    state["pontuacao_final"] = (
        state["pontuacao_relevancia"] * 0.3 +
        state["pontuacao_gramatica"] * 0.2 +
        state["pontuacao_estrutura"] * 0.2 +
        state["pontuacao_profundidade"] * 0.3
    )
    return state

# Inicializa o StateGraph
workflow = StateGraph(State)

# Adiciona nós ao grafo
workflow.add_node("verificar_relevancia", verificar_relevancia)
workflow.add_node("verificar_gramatica", verificar_gramatica)
workflow.add_node("analizar_estrutura", analizar_estrutura)
workflow.add_node("avaliar_profundidade", avaliar_profundidade)
workflow.add_node("calcular_pontuacao_final", calcular_pontuacao_final)

# Define e adiciona arestas condicionais
workflow.add_conditional_edges(
    "verificar_relevancia",
    lambda x: "verificar_gramatica" if x["pontuacao_relevancia"] > 0.5 else "calcular_pontuacao_final"
)
workflow.add_conditional_edges(
    "verificar_gramatica",
    lambda x: "analizar_estrutura" if x["pontuacao_gramatica"] > 0.6 else "calcular_pontuacao_final"
)
workflow.add_conditional_edges(
    "analizar_estrutura",
    lambda x: "avaliar_profundidade" if x["pontuacao_estrutura"] > 0.7 else "calcular_pontuacao_final"
)
workflow.add_conditional_edges(
    "avaliar_profundidade",
    lambda x: "calcular_pontuacao_final"
)

# Define o ponto de entrada
workflow.set_entry_point("verificar_relevancia")

# Define o ponto de saída
workflow.add_edge("calcular_pontuacao_final", END)


# Compila o grafo
app = workflow.compile()


def avaliar_redacao(redacao: str) -> dict:
    """Avalia a redação fornecida usando o fluxo de trabalho definido."""
    initial_state = State(
        redacao=redacao,
        pontuacao_relevancia=0.0,
        pontuacao_gramatica=0.0,
        pontuacao_estrutura=0.0,
        pontuacao_profundidade=0.0,
        pontuacao_final=0.0
    )
    result = app.invoke(initial_state)
    return result

exemplo_redacao = """
O Impacto dos Agentes de Inteligência Artificial e Fluxos de Trabalho Agentes com Multi-ferramentas na Sociedade Moderna


A Inteligência Artificial (IA) tem experimentado avanços significativos nas últimas décadas, e uma de suas manifestações mais revolucionárias é o desenvolvimento de agentes de IA e fluxos de trabalho agentes com multi-ferramentas. Estes agentes autônomos, capazes de aprender, adaptar-se e tomar decisões independentes, estão transformando a maneira como interagimos com a tecnologia e influenciando diversos setores da sociedade. Este ensaio explora os efeitos profundos desses agentes na sociedade moderna, discutindo seus benefícios, aplicações práticas e os desafios que apresentam.

No campo da saúde, os agentes de IA estão desempenhando um papel crucial na melhoria dos cuidados aos pacientes e na eficiência operacional das instituições médicas. Sistemas avançados como o DeepMind Health, do Google, utilizam agentes de IA para analisar imagens médicas, detectando sinais precoces de doenças como câncer e retinopatia diabética com precisão impressionante. Além disso, agentes com multi-ferramentas podem integrar dados de diferentes fontes, como registros eletrônicos de saúde, resultados de exames laboratoriais e informações genômicas, para oferecer diagnósticos mais precisos e planos de tratamento personalizados.

Durante a pandemia de COVID-19, agentes de IA foram empregados para modelar a propagação do vírus, prever surtos e otimizar a distribuição de recursos médicos. Esses agentes ajudaram governos e instituições de saúde a tomar decisões informadas sobre medidas de contenção e alocação de vacinas, demonstrando o potencial dos fluxos de trabalho agentes em situações de crise.

No setor financeiro, os agentes de IA estão redefinindo a análise de risco, detecção de fraude e gestão de investimentos. Robôs consultores (robo-advisors) utilizam algoritmos de IA para fornecer aconselhamento financeiro personalizado, considerando objetivos individuais, tolerância ao risco e condições de mercado. Empresas como a BlackRock e Vanguard incorporam agentes de IA em suas estratégias de investimento, permitindo decisões mais ágeis e baseadas em dados.

Além disso, agentes de IA com fluxos de trabalho agentes podem interagir com múltiplas ferramentas financeiras, monitorando transações em tempo real, analisando padrões de comportamento e identificando atividades suspeitas. Isso não apenas protege os consumidores, mas também fortalece a integridade do sistema financeiro global.

A indústria de transporte está passando por uma transformação significativa com a introdução de veículos autônomos e sistemas inteligentes de gerenciamento de tráfego. Agentes de IA são o cérebro por trás de carros autônomos, interpretando dados de sensores, câmeras e radares para tomar decisões de direção seguras. Empresas como Waymo, Uber e Tesla estão na vanguarda dessa tecnologia, prometendo reduzir acidentes causados por erro humano e melhorar a eficiência das viagens.

Em cidades inteligentes, agentes de IA gerenciam semáforos adaptativos, sistemas de transporte público e serviços de compartilhamento de veículos. Por exemplo, em Singapura, agentes inteligentes coordenam o fluxo de tráfego em tempo real, reduzindo congestionamentos e emissões de gases poluentes.

Na educação, agentes de IA estão promovendo uma revolução no ensino e aprendizado. Plataformas educacionais inteligentes utilizam agentes para adaptar currículos às necessidades individuais dos estudantes. Esses sistemas analisam o desempenho, identificam áreas de dificuldade e ajustam o conteúdo para otimizar o aprendizado.

Por exemplo, o sistema Knewton utiliza agentes de IA para fornecer recomendações personalizadas de estudo, enquanto o Squirrel AI, na China, oferece tutoria adaptativa para milhões de estudantes. Esses agentes permitem que os educadores atendam às necessidades específicas de cada aluno, promovendo um aprendizado mais eficaz e inclusivo.

Na indústria e manufatura, os agentes de IA estão otimizando processos de produção, manutenção preditiva e gerenciamento da cadeia de suprimentos. Robôs industriais equipados com agentes inteligentes podem adaptar-se a diferentes tarefas, aprender com erros e colaborar com trabalhadores humanos em ambientes de fábrica.

A Siemens, por exemplo, implementa agentes de IA em suas fábricas para monitorar o desempenho de máquinas, prever falhas e programar manutenções antes que ocorram interrupções. Isso resulta em maior eficiência, redução de custos e melhoria na qualidade dos produtos.

No setor de serviços, agentes de IA estão revolucionando o atendimento ao cliente. Chatbots e assistentes virtuais utilizam processamento de linguagem natural para interagir com clientes, resolver problemas comuns e encaminhar questões mais complexas a agentes humanos. Isso melhora a experiência do cliente e permite que as empresas atendam a um volume maior de consultas com eficiência.

Empresas como a Amazon e a Apple implementam agentes de IA em seus serviços de suporte, oferecendo assistência 24 horas por dia e personalizando as interações com base no histórico do cliente.

Apesar dos benefícios significativos, a adoção de agentes de IA e fluxos de trabalho agentes apresenta desafios complexos. A automação avançada ameaça substituir empregos em diversos setores, desde manufatura até serviços profissionais. Estudos do Fórum Econômico Mundial indicam que milhões de empregos podem ser afetados, exigindo políticas robustas de requalificação e educação para preparar a força de trabalho para novas oportunidades.

Questões éticas também são centrais neste debate. A tomada de decisões autônomas por agentes de IA levanta preocupações sobre responsabilidade legal em casos de falhas ou decisões prejudiciais. Além disso, algoritmos podem perpetuar vieses existentes se treinados em dados não representativos, levando a discriminação em áreas como contratação, crédito e justiça criminal.

A privacidade dos dados é outra área crítica. Agentes de IA dependem de grandes quantidades de dados pessoais para aprender e operar eficientemente. Garantir que esses dados sejam coletados e utilizados de maneira ética, respeitando regulamentos como a Lei Geral de Proteção de Dados (LGPD) no Brasil, é essencial para manter a confiança do público.

A crescente dependência de agentes de IA também levanta preocupações sobre segurança cibernética. Sistemas autônomos podem ser vulneráveis a ataques, manipulações e falhas sistêmicas. Incidentes envolvendo veículos autônomos ou sistemas financeiros automatizados podem ter consequências significativas, destacando a necessidade de protocolos de segurança robustos.

Além disso, há o risco de dependência excessiva da tecnologia, onde habilidades humanas podem ser negligenciadas ou perdidas. É importante equilibrar a adoção de agentes de IA com o desenvolvimento contínuo das capacidades humanas e o pensamento crítico.

O futuro dos agentes de IA é promissor, com avanços contínuos em áreas como processamento de linguagem natural, aprendizado por reforço e redes neurais profundas. A integração de agentes de IA com tecnologias emergentes como computação quântica e Internet das Coisas (IoT) ampliará ainda mais suas capacidades.

Iniciativas como o AutoGPT e o BabyAGI representam passos em direção a agentes de IA capazes de aprender e se adaptar de forma autônoma, executando tarefas complexas sem intervenção humana constante. Esses sistemas têm o potencial de revolucionar setores inteiros, mas também exigem uma consideração cuidadosa dos impactos sociais e éticos.

Em conclusão, os agentes de IA e fluxos de trabalho agentes com multi-ferramentas estão remodelando a sociedade moderna, oferecendo soluções inovadoras para desafios complexos e melhorando a eficiência em diversos setores. Seus benefícios são inegáveis, desde cuidados de saúde aprimorados até transportes mais seguros e educação personalizada.

No entanto, esses avanços vêm acompanhados de desafios significativos que exigem atenção cuidadosa. A sociedade deve abordar questões de emprego, ética, privacidade e segurança de dados para garantir que a integração de agentes de IA beneficie a todos de maneira equitativa.

À medida que avançamos para um futuro cada vez mais interconectado e orientado pela IA, é responsabilidade coletiva de governos, empresas, educadores e cidadãos trabalhar juntos. Devemos promover uma abordagem equilibrada que valorize a inovação tecnológica ao mesmo tempo em que protege os valores humanos fundamentais, assegurando que os agentes de IA sirvam como ferramentas para o bem-estar e progresso da humanidade.
    """

# Avalia a redação de exemplo
result = avaliar_redacao(exemplo_redacao)

# Converte as pontuações de 0-1 para 0-10
pontuacao_final = result['pontuacao_final'] * 10
pontuacao_relevancia = result['pontuacao_relevancia'] * 10
pontuacao_gramatica = result['pontuacao_gramatica'] * 10
pontuacao_estrutura = result['pontuacao_estrutura'] * 10
pontuacao_profundidade = result['pontuacao_profundidade'] * 10

# Exibe os resultados
print(f"Pontuação Final da Redação: {pontuacao_final:.2f}/10\n")
print(f"Pontuação de Relevância: {pontuacao_relevancia:.2f}/10")
print(f"Pontuação de Gramática: {pontuacao_gramatica:.2f}/10")
print(f"Pontuação de Estrutura: {pontuacao_estrutura:.2f}/10")
print(f"Pontuação de Profundidade: {pontuacao_profundidade:.2f}/10")