# RAG-basierter Chatbot für IT Service Management

Diese Repository enthält die vollständige Implementierung eines RAG-basierten (Retrieval-Augmented Generation) Chatbots, der im Rahmen einer Masterarbeit zum Einsatz von Chatbots im IT Service Management entwickelt wurde. Das System demonstriert den praktischen Nutzen von KI-gestützten Assistenten in Serviceorganisationen und bietet sowohl ein Basismodell als auch eine erweiterte Implementierung mit fortgeschrittenen Zuverlässigkeitsmechanismen.

## Überblick

Das Repository präsentiert zwei distinct Entwicklungsstufen eines Chatbot-Systems, das speziell für den Einsatz in PLM-Umgebungen (Product Lifecycle Management) konzipiert wurde. Die Implementierung nutzt lokale Sprachmodelle zur Gewährleistung von Datenschutz und organisatorischer Autonomie, während gleichzeitig moderne RAG-Techniken für präzise und kontextuelle Antworten eingesetzt werden.

Die erste Implementierung stellt ein funktionsfähiges Basismodell dar, das die grundlegenden RAG-Prinzipien demonstriert und eine solide Grundlage für praktische Anwendungen bietet. Die zweite Iteration erweitert das System um ausgeklügelte Mechanismen zur Bewertung der Antwortqualität, neuronale Dokumenten-Neuordnung und erweiterte Abrufverfahren, die eine signifikante Verbesserung der Systemzuverlässigkeit ermöglichen.

## Systemarchitektur

Das System basiert auf einer modularen Architektur, die LLama 3 8B als lokales Sprachmodell über Ollama integriert. Die Dokumentenverarbeitung erfolgt durch das LangChain-Framework, das eine flexible und erweiterbare Grundlage für komplexe RAG-Workflows bietet. ChromaDB dient als primäre Vektordatenbank für die persistente Speicherung von Dokumenten-Embeddings, während HuggingFace Sentence Transformers für die Generierung semantischer Repräsentationen eingesetzt werden.

Die Benutzeroberfläche wird durch Gradio bereitgestellt, das eine intuitive webbasierte Interaktion ermöglicht und sowohl für Demonstrationszwecke als auch für produktive Anwendungen geeignet ist. Die gesamte Architektur ist darauf ausgelegt, vollständig lokal betrieben zu werden, wodurch sensitive Unternehmensdaten geschützt bleiben und keine Abhängigkeiten von externen API-Diensten entstehen.

## Implementierungen

### Basismodell (chatbot_base_model.py)

Das Basismodell implementiert eine straightforward RAG-Pipeline, die dokumentenbasierte Frage-Antwort-Funktionalität mit einer einfachen und nachvollziehbaren Systemarchitektur verbindet. Die Implementierung lädt Dokumente aus einer konfigurierbaren Wissensbasis, verarbeitet diese in semantisch sinnvolle Chunks und erstellt einen durchsuchbaren Vektorindex.

Das System nutzt Konversationsgedächtnis zur Aufrechterhaltung des Dialogkontexts und implementiert domänenspezifische Richtlinien für PLM-bezogene Anfragen. Die Retrieval-Komponente ist so konfiguriert, dass sie die 25 relevantesten Dokumentenfragmente für jede Anfrage berücksichtigt, wodurch eine umfassende Kontextabdeckung gewährleistet wird.

### Erweiterte Implementierung (chatbot_iteration2.py)

Die erweiterte Implementierung erweitert das Basismodell um mehrere sophistizierte Mechanismen zur Qualitätssicherung und Leistungsoptimierung. Das System implementiert automatische Dokumentenrelevanz-Prüfung, um sicherzustellen, dass nur thematisch passende Informationen für die Antwortgenerierung verwendet werden.

Ein integriertes Halluzinations-Erkennungssystem bewertet kontinuierlich, ob generierte Antworten durch die abgerufenen Dokumente gestützt werden. Neuronales Re-ranking durch Cross-Encoder verbessert die Dokumentenauswahl erheblich, während Contextual Compression irrelevante Informationen aus abgerufenen Dokumenten entfernt.

Das System bietet optional Hypothetical Prompt Embeddings (HyPE), eine fortgeschrittene Technik, die die Abrufgenauigkeit durch die Generierung hypothetischer Fragen für jeden Dokumenten-Chunk verbessert. Ein umfassendes Konfidenz-Bewertungssystem informiert Benutzer über die Zuverlässigkeit jeder generierten Antwort.

## Installation und Einrichtung

Die Installation erfordert Python 3.8 oder höher sowie die separate Installation von Ollama für die lokale LLM-Integration. Erstellen Sie zunächst eine virtuelle Python-Umgebung und aktivieren Sie diese, um Abhängigkeitskonflikte zu vermeiden.

Installieren Sie die erforderlichen Python-Pakete durch Ausführung von `pip install -r requirements.txt` im Projektverzeichnis. Laden Sie Ollama von der offiziellen Website herunter und installieren Sie es entsprechend den Anweisungen für Ihr Betriebssystem.

Nach der Ollama-Installation führen Sie `ollama pull llama3:8b` aus, um das erforderliche Sprachmodell herunterzuladen. Bereiten Sie Ihre Wissensdatenbank vor, indem Sie Dokumente im Markdown-Format in dem entsprechenden Verzeichnis ablegen.

## Verwendung

Starten Sie das Basismodell durch Ausführung von `python chatbot_base_model.py` oder die erweiterte Version mit `python chatbot_iteration2.py`. Das System startet automatisch eine lokale Weboberfläche, die über Ihren Browser zugänglich ist.

Die erweiterte Implementierung bietet zusätzliche Konfigurationsoptionen für verschiedene Retrieval-Methoden und Qualitätssicherungsmechanismen. Benutzer können zwischen verschiedenen Re-ranking-Algorithmen wählen und optionale Features wie Contextual Compression oder HyPE aktivieren.

## Anpassung für eigene Anwendungsfälle

Das System ist modular aufgebaut und kann für verschiedene Domänen und Anwendungsfälle angepasst werden. Ersetzen Sie die Beispiel-Wissensdatenbank durch Ihre eigenen Dokumente und passen Sie die domänenspezifischen Prompts in den Konfigurationsbereichen an.

Die Chunk-Größe und Überlappungsparameter können je nach Dokumententyp und gewünschter Granularität angepasst werden. Für spezialisierte Anwendungsfälle können alternative Embedding-Modelle oder zusätzliche Dokumentenverarbeitungs-Pipeline implementiert werden.

## Forschungskontext

Diese Implementierung entstand im Rahmen einer Masterarbeit, die die praktische Anwendbarkeit von Chatbots in IT Service Management-Umgebungen untersuchte. Die Forschung konzentrierte sich auf die Evaluierung verschiedener technologischer Ansätze und deren Einfluss auf die Antwortqualität und Benutzerzufriedenheit.

Die vergleichende Analyse zwischen Basis- und erweiterter Implementierung lieferte wichtige Erkenntnisse über die relative Bedeutung von Wissensmanagement-Praktiken gegenüber rein technologischen Verbesserungen. Die Ergebnisse zeigen, dass strukturiertes Wissensmanagement oft größeren Einfluss auf die Systemeffektivität hat als sophisticated technische Optimierungen.

## Technische Dokumentation

Ausführliche technische Dokumentation finden Sie in den begleitenden Markdown-Dateien, die Installationsanweisungen, Abhängigkeitsbeschreibungen und Konfigurationsoptionen detailliert erläutern. Die Implementierung folgt etablierten Software-Engineering-Praktiken und ist vollständig dokumentiert.

## Lizenz und Verwendung

Dieses Projekt steht unter der MIT-Lizenz und kann frei für Forschungs- und Bildungszwecke verwendet werden. Bei wissenschaftlicher Nutzung wird um entsprechende Zitation gebeten. Kommerzielle Anwendungen sollten die Lizenzbedingungen der verwendeten Bibliotheken beachten.

## Beiträge und Weiterentwicklung

Beiträge zur Verbesserung und Erweiterung des Systems sind willkommen. Öffnen Sie Issues für Fehlermeldungen oder Feature-Anfragen und reichen Sie Pull Requests für Verbesserungen ein. Besonderes Interesse besteht an Erweiterungen für zusätzliche Domänen und Verbesserungen der Evaluationsmechanismen.

## Kontakt

Für Fragen zur Implementierung oder zum Forschungskontext können Sie über die GitHub-Issue-Funktion Kontakt aufnehmen. Detaillierte Diskussionen über die zugrundeliegende Forschung und deren Ergebnisse sind in der entsprechenden Masterarbeit dokumentiert.
