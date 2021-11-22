# PBB
Python Skript &amp; Daten für einen Aufsatz in PBB zur Produktivität und zum Rückbau von -nis

Das Python-Skript nis_monte_carlo_PBB_github.py braucht zwei Datensätze:
1. Eine Liste mit allen Veröffentlichungen im Deutschen Textarchiv und im Kernkorpus des Digitalen Wörterbuchs der Deutschen Sprache mit Dateinamen, Jahr und Länge in Wörtern (texts_dta_dwds_5000.csv)
2. Eine Liste mir allen -nis-Bildungen inklusive Ursprungsdatei (nis_re-lemmatized_total.csv).

Es benötigt außerdem die Datei monte_carlo.py, die das Multiprocessing regelt.

Das Skript benötigt folgende Pakete: 
pandas, numpy, re, itertools, timeit
