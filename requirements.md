# Installationsanforderungen und Abhängigkeiten

Dieses Dokument beschreibt die Software-Abhängigkeiten und Installationsanforderungen für die RAG-basierte Chatbot-Implementierung, die im Rahmen der Masterarbeit zu IT Service Management Anwendungen entwickelt wurde.

## Systemvoraussetzungen

Die Implementierung erfordert Python 3.8 oder höher und ist mit Windows-, macOS- und Linux-Betriebssystemen kompatibel. Benutzer sollten sicherstellen, dass der pip-Paketmanager installiert und auf die neueste Version aktualisiert ist, bevor sie mit der Installation fortfahren.

## Grundlegende Framework-Abhängigkeiten

Die Chatbot-Implementierung basiert auf dem LangChain-Framework als primäre Orchestrierungsschicht. LangChain Version 0.1.20 stellt die grundlegenden Komponenten für Dokumentenladung, Textverarbeitung und Verkettungsoperationen bereit. Die Implementierung benötigt außerdem langchain-core Version 0.1.52 für wesentliche Abstraktionen und langchain-community Version 0.0.38 für von der Community beigesteuerte Integrationen. Die Vektorspeicher-Funktionalität hängt von langchain-chroma Version 0.1.2 für die nahtlose Integration mit der ChromaDB-Vektordatenbank ab.

## Integration lokaler Sprachmodelle

Das System integriert lokale Sprachmodelle über Ollama Version 0.1.7, das eine optimierte Schnittstelle für die lokale Ausführung von LLama 3 8B bereitstellt. Dieser Ansatz gewährleistet Datenschutz und eliminiert Abhängigkeiten von externen API-Diensten. Benutzer müssen Ollama separat installieren und das LLama 3 8B Modell über die entsprechenden Ollama-Befehle herunterladen.

## Embedding- und Vektorspeicher-Komponenten

Die Dokumenten-Embedding-Funktionalität erfordert sentence-transformers Version 2.7.0 für die Generierung hochwertiger semantischer Embeddings. Die Vektorspeicher-Schicht nutzt ChromaDB Version 0.4.24 für persistente Speicherung und FAISS-CPU Version 1.8.0 für effiziente Ähnlichkeitssuchoperationen. Diese Komponenten arbeiten zusammen, um robuste Dokumentenabruf-Fähigkeiten bereitzustellen.

## Machine Learning- und Transformer-Bibliotheken

Die erweiterte Iteration integriert Hugging Face transformers Version 4.40.2 für lokale Modellintegration und erweiterte Textverarbeitungsfähigkeiten. PyTorch Version 2.0.0 oder höher stellt die zugrundeliegenden Tensor-Operationen und neuronalen Netzwerk-Funktionalitäten bereit. Die tokenizers-Bibliothek Version 0.19.1 gewährleistet effiziente Texttokenisierung für verschiedene Modellarchitekturen.

## Neural Re-ranking und erweiterte Abrufverfahren

Die zweite Iteration implementiert neuronales Re-ranking durch die cross-encoder-Bibliothek Version 1.2.0, die eine ausgeklügelte Dokumentenrelevanz-Bewertung ermöglicht. Diese Komponente verbessert die Abrufgenauigkeit erheblich, indem sie kontextbewusstes Ranking von Kandidatendokumenten bereitstellt.

## Benutzeroberfläche und Web-Framework

Die Chatbot-Oberfläche nutzt Gradio Version 4.29.0, um eine intuitive webbasierte Interaktionsschicht bereitzustellen. Gradio ermöglicht schnelles Prototyping von Machine Learning-Interfaces und unterstützt sowohl lokale Bereitstellung als auch Sharing-Funktionen für Demonstrationszwecke.

## Dokumentenverarbeitung und Datenbehandlung

Die Dokumentenverarbeitungsfähigkeiten umfassen Unterstützung für mehrere Dateiformate durch pypdf Version 4.2.0 für PDF-Dokumente, python-docx Version 1.1.0 für Microsoft Word-Dateien und markdown Version 3.6 für Markdown-Textverarbeitung. Datenmanipulation und -analyse basieren auf numpy Version 1.24.4 und pandas Version 2.0.3 für effiziente numerische Operationen und strukturierte Datenbehandlung.

## Utility- und Support-Bibliotheken

Fortschrittsverfolgung während Dokumentenverarbeitung und Modelloperationen verwendet tqdm Version 4.66.4 für benutzerfreundliche Fortschrittsbalken. Konfigurationsverwaltung nutzt python-dotenv Version 1.0.1 für Umgebungsvariablen-Behandlung. Typvalidierung und Datenstrukturdefinitionen verwenden pydantic Version 2.7.1 zusammen mit typing-extensions Version 4.11.0 für erweiterte Typsicherheit.

## Visualisierungs- und Analyse-Tools

Die Implementierung umfasst matplotlib Version 3.7.5 und plotly Version 5.20.0 für die Erstellung von Visualisierungen und Leistungsanalyse-Diagrammen. Diese Bibliotheken unterstützen die Evaluations- und Demonstrationsphasen der Forschungsimplementierung. Scikit-learn Version 1.4.2 stellt zusätzliche Machine Learning-Utilities für Dokumentenähnlichkeitsberechnungen und Clustering-Operationen bereit.

## Optionale erweiterte Funktionen

Benutzer mit GPU-Hardware können optional bitsandbytes Version 0.43.1 und accelerate Version 0.30.1 für Modellquantisierung und Hardware-Beschleunigung installieren. Diese Abhängigkeiten ermöglichen effizienteren Speicherverbrauch und schnellere Inferenzzeiten bei der Arbeit mit größeren Sprachmodellen.

Erweiterte Textverarbeitungsfähigkeiten können durch optionale Installation von spaCy Version 3.7.4 und NLTK Version 3.8.1 erreicht werden. Diese Bibliotheken bieten fortgeschrittene Funktionen zur Verarbeitung natürlicher Sprache wie Named Entity Recognition und linguistische Analyse.

## Entwicklung und Qualitätssicherung

Entwicklungsumgebungen können von der Einbeziehung von pytest Version 8.2.1 für umfassende Tests, black Version 24.4.2 für konsistente Code-Formatierung und flake8 Version 7.0.0 für Code-Qualitätsanalyse profitieren. Diese Tools gewährleisten wartbaren und zuverlässigen Code während des gesamten Entwicklungsprozesses.

## Installationsanweisungen

Erstellen Sie eine virtuelle Umgebung, um die Projektabhängigkeiten zu isolieren, und aktivieren Sie diese vor der Installation. Installieren Sie die Kernabhängigkeiten mit pip über den Befehl `pip install -r requirements.txt`. Stellen Sie sicher, dass Ollama separat installiert wird, indem Sie der offiziellen Installationsanleitung für Ihr Betriebssystem folgen. Laden Sie das LLama 3 8B Modell über Ollama mit den entsprechenden Modell-Download-Befehlen herunter.

## Versionskompatibilitäts-Hinweise

Die angegebenen Bibliotheksversionen wurden auf Kompatibilität und stabilen Betrieb in der Forschungsumgebung getestet. Während neuere Versionen zusätzliche Funktionen oder Leistungsverbesserungen bieten können, sollten Benutzer, die Kompatibilitätsprobleme erfahren, zu den in dieser Dokumentation angegebenen exakten Versionen zurückkehren. Regelmäßige Abhängigkeits-Updates sollten gründlich in einer Entwicklungsumgebung getestet werden, bevor sie in Produktionssystemen eingesetzt werden.

## Hardware-Empfehlungen

Die Basismodell-Implementierung funktioniert effizient auf Standard-Consumer-Hardware mit mindestens 8GB RAM. Die erweiterte Iteration mit Neural Re-ranking und fortgeschrittenen Funktionen profitiert von 16GB RAM oder höher. GPU-Beschleunigung ist optional, aber empfohlen für Benutzer, die große Dokumentensammlungen verarbeiten oder schnellere Inferenzzeiten benötigen.