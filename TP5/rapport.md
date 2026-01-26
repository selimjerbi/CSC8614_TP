# TP5

## Exercice 1

![alt text](img/q1.png)

## Exercice 2

![alt text](img/liste_mails.png)

Le jeu de test est composé de 10 emails couvrant des cas administratifs, pédagogiques et de recherche. Il inclut également un email ambigu nécessitant une clarification et un email à risque simulant une demande sensible. Cette diversité permet de tester le parsing, la détection d’intentions et la robustesse du système face à des contenus variés.

10 emails chargés : 

![alt text](img/load_mails.png)

## Exercice 3

![alt text](img/run_json.png)

## Exercice 4

![alt text](img/sortie_test_router.png) 

![alt text](img/run_router.png)

## Exercice 5 

![alt text](img/version.png)


![alt text](img/sortie_graph.png)


![alt text](img/graph_json.png)

## Exercice 6

![alt text](img/sortie_ex6.png)


![alt text](img/json_ex6.png) 

## Exercice 7

![alt text](img/mode_prud.png) 


![alt text](img/json_ex7.png)

## Exercice 8

![alt text](img/modif_ex8.png)


![alt text](img/sortie_ex8.png)

## Exercice 9 

![alt text](img/modif_ex9.png)

![alt text](img/version_final_e01.png) 

![alt text](img/mail_ignoré_ex9.png) 

![alt text](img/json_ex9.png)

## Exercice 10

![alt text](img/sortie_ex10.png)

## Exercice 11 

![alt text](img/sortie_ex11.png)

![alt text](img/bacth_res.png)

Sur les 11 emails testés, l’intent largement dominant est **reply** (9 emails sur 11),
ce qui correspond à des demandes classiques d’information ou de clarification.
Une seule escalade est observée (**E11 – Test prompt injection**), déclenchée par
l’heuristique de détection de prompt injection, avec un `risk_level=high` et
un `final_kind=handoff`, sans appel à des tools hors périmètre.
Aucun safe mode explicite n’est activé dans cette configuration, mais l’escalade
joue un rôle équivalent de protection.
La majorité des emails suivent une trajectoire simple avec **1 tentative de retrieval**
et **1 appel tool**, conduisant directement à une réponse finale.
Un cas intéressant est l’email **E11**, où malgré une tentative de récupération initiale,
l’agent force l’escalade et empêche toute action dangereuse, illustrant l’efficacité
des garde-fous déterministes face à une attaque par prompt injection.



**Run simple (E08)**  

![alt text](img/run_08.png)

La trajectoire débute par une classification qui conclut à un intent
de réponse standard. L’agent effectue ensuite une unique étape de retrieval,
déclenchant un seul appel au tool `rag_search`.
Après la récupération des documents, l’agent passe directement à la génération
de la réponse sans vérification supplémentaire ni boucle.
La trajectoire est courte, linéaire, et se termine sans escalade ni mécanisme de sûreté.


**Run complexe (E10)**  

![alt text](img/run_10.png)

Après la classification initiale, l’agent engage une phase de retrieval
avec appel au tool `rag_search`. Contrairement au run simple, la génération de réponse
est suivie d’une étape explicite de vérification des preuves, avant la finalisation.
Cette trajectoire illustre un routage plus riche avec contrôle intermédiaire de la qualité
des sources, sans toutefois déclencher de boucle ni d’escalade.
Elle montre comment l’agent adapte son cheminement en fonction du contexte et du besoin
de justification.

## Exercice 12

# 1. Exécution
Réponse RAG simple (TP5)
python TP5/rag_answer_tp5.py "Votre question ici"

Test du graphe minimal sur un email
python -m TP5.test_graph_minimal

Exécution batch sur 8–12 emails
python -m TP5.run_batch

![alt text](img/sortie_ex7.png)

![alt text](img/sortie_ex10.png)

# 2. Architecture

flowchart TD
    A[classify_email]
    B[maybe_retrieve]
    C[rewrite_query]
    D[check_evidence]
    E[draft_reply]
    F[finalize]

    A -->|needs_retrieval = true| B
    A -->|intent = escalate / ignore| F

    B -->|no docs / retry| C
    C --> B

    B -->|docs found| E
    E --> D
    D --> F

# Rôle des nœuds
classify_email : analyse l’email, décide de l’intent, du niveau de risque et de l’accès au retrieval.
maybe_retrieve : effectue la recherche documentaire (RAG) sous contraintes de budget et d’allow-list.
rewrite_query : reformule la requête en cas de retrieval insuffisant.
draft_reply : génère une réponse candidate à partir du contexte récupéré.
check_evidence : vérifie la cohérence et la présence de sources pertinentes.
finalize : produit soit une réponse finale, soit un handoff sécurisé en cas de risque élevé.

# 3. Résultats & 4. Trajectoires :
voir exercice 11

# 5. Réflexion finale

Ce qui marche bien :
L’instrumentation fine (logs JSONL, batch, trajectoires) permet une observabilité complète.
Les garde-fous déterministes (heuristique, budgets, allow-list) protègent efficacement
contre les comportements dangereux, indépendamment du LLM.

Ce qui est fragile :
Les heuristiques de prompt injection restent basées sur des motifs simples.
Le parsing des emails est strict et dépend fortement du format des données.
Amélioration prioritaire :
Ajouter une détection de risque hybride (heuristique + score LLM léger)
et un comptage structuré des métriques (steps, retries, tools) sans parsing texte des logs.

Conclusion
Ce TP montre qu’un agent fiable ne se limite pas à produire une réponse,
mais repose sur une architecture contrôlée, des preuves d’exécution,
et une évaluation par trajectoires, en ligne avec des pratiques MLOps/LLMOps réalistes.
