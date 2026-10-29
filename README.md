# Rede Neural 3D

Uma demonstração educativa de uma rede neural simples aplicada ao problema XOR, com visualização gráfica em 3D baseada em grafos e nós.

## Modelo

O modelo Python implementa:

- 2 nós de entrada;
- 6 nós ocultos com ativação `tanh`;
- 1 saída com ativação `sigmoid`;
- treinamento por retropropagação e gradiente descendente;
- representação dos pesos como arestas de um grafo.

O projeto não exige bibliotecas Python externas.

## Executar

```bash
python3 app.py
```

Acesse `http://localhost:8000`.

## Controles

- Arraste a cena com o mouse para girar.
- Use a roda do mouse para aproximar ou afastar.
- Altere X e Y para enviar novos valores ao modelo.
