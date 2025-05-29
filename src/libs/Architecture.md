Grobe Architektur (Idee):

- lib/LLMCLient
    - Regelt Kommunikation zwischen LLM <-> Außenwelt
    - Jedes Module spawnt dabei ein eigenen LLM-Lifecycle
        - Dadurch haben wir die Möglichkeit Agents in einem Lifecycle zu verwenden
        - Erst wenn der LLM-Lifecycle abgeschlossen ist returnen wir ein Result
        - Könnte so aussehen:
            - Init
            - Action / Agent / Tools Loop
            - Result
- pipeline/LLMPipeline
    - Hat LLMClient
    - Übergibt LLMClient an Modules
- modules/llm/*
    - Nutzen LLMCLient
    - Enthalten ihre Prompts
    - Enthalten ihre Tools
    - Enthalten ihre gewünschten Returns
    - Führen kompletten LLM-Lifecycle aus
