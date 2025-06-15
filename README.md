# RAG-basierter Chatbot im IT Service Management

Dieses Repository dokumentiert die Implementierung eines RAG-basierten Chatbots, der im Rahmen einer Masterarbeit zum Einsatz von KI im IT Service Management entwickelt wurde. Die Implementierung demonstriert zwei Entwicklungsstufen eines lokalen Chatbot-Systems.

## Überblick der Implementierungen

Das Repository enthält zwei implementierte Chatbot-Versionen, die verschiedene technologische Ansätze demonstrieren. Beide Implementierungen nutzen LLama 3 8B als lokales Sprachmodell über Ollama und vermeiden externe API-Abhängigkeiten zur Gewährleistung des Datenschutzes.

### Basismodell (chatbot_base_model.py)

Das Basismodell implementiert eine grundlegende RAG-Pipeline mit LangChain-Framework und ChromaDB-Vektorspeicherung. Das System lädt Dokumente aus einer Wissensdatenbank, verarbeitet diese in semantische Chunks und erstellt einen durchsuchbaren Index. Die Implementierung nutzt HuggingFace Sentence Transformers für lokale Embeddings und Gradio für die webbasierte Benutzeroberfläche.

### Erweiterte Implementierung (chatbot_iteration2.py)

Die erweiterte Version ergänzt das Basismodell um Mechanismen zur Qualitätssicherung und Zuverlässigkeitsbewertung. Das System implementiert automatische Prüfungen der Relevanz, Halluzinationserkennung und neuronales Re-ranking durch Cross-Encoder. Es stehen darüber hinaus Hypothetical Prompt Embeddings (HyPE) und Contextual Compression zur Verfügung. Ein integriertes Konfidenzscore-System bewertet die Zuverlässigkeit der generierten Antworten.

## Technische Architektur

Die modulare Systemarchitektur basiert auf Open-Source-Komponenten. LangChain orchestriert die RAG-Pipeline, während ChromaDB die persistente Vektorspeicherung übernimmt. Die Dokumentenverarbeitung erfolgt durch konfigurierbare Text-Splitter mit anpassbaren Chunk-Größen und Überlappungsparametern. Alle Komponenten sind für lokalen Betrieb konzipiert und erfordern keine externen Cloud-Services.

## Installation und Ausführung

Die Ausführung erfordert Python 3.8 oder höher sowie eine separate Ollama-Installation für die LLM-Integration. Nach der Installation der Python-Abhängigkeiten über `pip install -r requirements.txt` und dem Download des LLama 3 8B Modells über `ollama pull llama3:8b` können die Implementierungen direkt gestartet werden.

## Forschungskontext und Erkenntnisse

Die Implementierung entstand zur Untersuchung der praktischen Anwendbarkeit von Chatbots in IT Service Management-Umgebungen. Die vergleichende Evaluierung beider Versionen lieferte Erkenntnisse über die relative Bedeutung von Wissensmanagement-Praktiken gegenüber technologischen Optimierungen. Die Forschungsergebnisse zeigen, dass strukturiertes Wissensmanagement häufig größeren Einfluss auf die Systemeffektivität ausübt als ausschließlich technische Verbesserungen.

## Dokumentation und Nutzung

Die Repository-Struktur umfasst vollständige Abhängigkeitslisten, Installationsanleitungen und technische Dokumentation. Die Implementierungen dienen primär zur Demonstration der entwickelten Konzepte und Dokumentation der Forschungsergebnisse.
