"""
Prompt templates for DiKoBi coding manual categories.
Each category has a specific prompt template and valid output range.

Reference: DiKoBi Coding Manual
https://epub.ub.uni-muenchen.de/77972/1/MCLSReports_Kodiermanual.pdf

Note: Placeholders for categories 2-27 should be filled in with actual DiKoBi definitions.
Category 1_D_M is fully implemented as reference.

Prompt variations are defined in PROMPT_TEMPLATES for experimentation.
"""

from ..preprocessing.standards import MAX_VALUES

# System prompt used across all categories
SYSTEM_PROMPT = """You are an expert human coder in biology education research.

Task: Assign ONE integer rating for the given category ONLY.
Use ONLY the category definition and rating scale provided below.
Do NOT infer or score any other categories/aspects.

Output format: Reply with exactly one line in the form "Rating: X" where X is a valid integer rating.
If the text is unrelated to the category or provides no evidence, return "Rating: 0".
Do NOT include any other text."""

# JSON-based system prompt (more reliable for some models)
SYSTEM_PROMPT_JSON = """You are an expert human coder in biology education research.

Task: Assign ONE integer rating for the given category ONLY.
Use ONLY the category definition and rating scale provided below.
Do NOT infer or score any other categories/aspects.

Output format: Respond with valid JSON in this exact format:
{
  "rating": X,
  "explanation": "brief justification (1-2 sentences)"
}

Where X is a valid integer rating.
If the text is unrelated to the category or provides no evidence, return rating 0.
Your response must be valid JSON only - no other text before or after."""

# TODO: Add few-shot examples for other categories (follow 1_D_M structure)
# TODO: Implement RAG (Retrieval-Augmented Generation) for dynamic example selection
#       RAG would retrieve most relevant examples from a database based on similarity
#       to the input text, enabling better context-aware classification.


def get_system_prompt(use_json: bool = False) -> str:
    """Get the appropriate system prompt based on output format.
    
    Args:
        use_json: If True, returns JSON-based prompt. If False, returns text-based prompt.
        
    Returns:
        System prompt string
    """
    return SYSTEM_PROMPT_JSON if use_json else SYSTEM_PROMPT


# Category-specific definitions and rating scales
CATEGORIES = {
    "1_D_M": {
        "name": "Unterrichtseinstieg - Description - Motivation",
    "definition": """Beschreiben (Description): Fehlende Motivation / fehlendes Interesse im Unterrichtseinstieg.

Kodiert wird, ob die Antwort einen verbesserungsfähigen Aspekt des Einstiegs beschreibt, der darauf abzielt, die momentane Lernmotivation bzw. das (situationale) Interesse für das neue Thema zu wecken.

Typische inhaltliche Aspekte (Beispiele):
- fehlende Anschaulichkeit / fehlende Beispiele
- fehlende Ich-Nähe / Alltagsbezug
- fehlender Anwendungsbezug / Kontext
- fehlende (catch-)Komponente des situationalen Interesses (z.B. Überraschung, Diskrepanz)
- fehlende Anreize (sachbezogener Anreiz), fehlende Aufmerksamkeit/Zuwendung
- fehlende neuartige/überraschende Inhalte

Nicht kodieren (andere Kategorien):
- reine Begrüßung/Sozialform/Methoden ohne Motivationsbezug (PK)
- kognitive Aktivierung ohne Motivationsbezug (1_D_kA)

Wenn der beschriebene Aspekt nicht zu diesem fachdidaktischen Schwerpunkt gehört, ist 0 zu kodieren.""",
        
        "rating_scale": {
            0: """(trifft nicht zu) - Does not apply
• No mention of motivational aspects at all
• Only mentions other aspects (greeting, cognitive activation, duration, teaching methods)
• Too generic/vague to categorize""",
            
            1: """(Unsystematisches/unvollständiges Beschreiben) - Incomplete/unsystematic description
• General mention that motivation is missing, without concrete details
• Uses theoretical keywords (catch-Komponente, Kontext, Interesse) but lacks specific description
• States WHAT is missing but not HOW it manifests
• Single aspect mentioned without elaboration""",
            
            2: """(Systematisches/vollständiges Beschreiben) - Complete/systematic description
• Describes WHAT specifically is missing AND provides details/attributes
• Strong focus on observable, concrete details from the video
• Includes descriptive attributes (e.g., "quickly," "briefly," "not exciting")
• May mention multiple related aspects
• Contains enough specificity that someone could understand the scene"""
        },
        
        # Few-shot examples for this category (manually selected from train CSV)
        "examples": [
            {"text": "Kein motivationaler Einstieg - fehlende catch komponente", "rating": 2},
            {"text": "zwar bezug auf letzte Stunde aber recht wenig motivation für die Kinder", "rating": 1},
            {"text": "Körperhaltung", "rating": 0},
        ]
    },
    
    # ============================================================================
    # PLACEHOLDER CATEGORIES - Fill in with DiKoBi manual definitions
    # ============================================================================
    # TODO: Add definition, rating_scale, and examples like 1_D_M above
    
    # Topic 1: Unterrichtseinstieg (Lesson Introduction)
    "1_D_kA": {
        "name": "Unterrichtseinstieg - Beschreiben - kognitive Aktivierung",
        "definition": """Beschreiben (Description): Fehlende kognitive Aktivierung im Unterrichtseinstieg.

Kodiert wird, ob die Antwort einen verbesserungsfähigen Aspekt des Einstiegs beschreibt, der (aus fachdidaktischer Perspektive) unzureichend kognitiv aktivierend ist.

Typische Aspekte (Beispiele):
- Einstieg ist nur eine kurze/oberflächliche Wiederholung (reine Reproduktion; nur Begriffe aufzählen)
- keine Problemorientierung / keine Fokusfrage
- kein kognitiver Konflikt
- Fragestellungen sind rein reproduktiv; keine Erklärungen werden eingefordert
- Thema/Ziel wird von der Lehrkraft vorgegeben; SuS formulieren Ziel nicht selbst

Nicht kodieren:
- reine Zeitangaben („Einstieg kurz“) ohne Bezug auf kognitive Prozesse (häufig PK)
- unspezifische Aussagen („fehlende Hinführung“) ohne Zuordnung

Wenn der Aspekt nicht zu diesem fachdidaktischen Schwerpunkt gehört, ist 0 zu kodieren.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description vorhanden ODER
- kein Bezug zu kognitiver Aktivierung im Einstieg (anderer Aspekt)
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- unspezifisch/oberflächlich, wenig Detail
- oft ein Schlagwort zur Zuordnung (z.B. „keine Problemorientierung“, „nur Wiederholung“)
""",
            2: """Systematisches / vollständiges Beschreiben
- mehrere spezifische Details, sichtbar im Video
- mit Attributen/konkreten Merkmalen (z.B. wie die Wiederholung abläuft, welche Fragen gestellt werden)
""",
        },
        "examples": [
            {"text": "Das Unterrichtsziel wird von der Lehrkraft klar vorgegeben und nicht mit den Schülern erarbeitet", "rating": 2},
            {"text": "Aktivieren des Vorwissens nur durch reine Reproduktion", "rating": 1},
            {"text": "fehlende Begrüßung", "rating": 0},
        ],
    },
    "1_Dm_M": {
        "name": "Unterrichtseinstieg - Beschreiben/Merken - Motivation",
        "definition": """Decision Making (Dm): Handlungsalternative zur Verbesserung der Motivation / des situationalen Interesses im Unterrichtseinstieg.

Kodiert wird, ob die Antwort beschreibt, wie die Lehrkraft (aus fachdidaktischer Perspektive) anders handeln würde, um die zuvor kritisierten Motivations-/Interessaspekte zu verbessern (z.B. Catch-Komponente, Alltagsbezug, Ich-Nähe, Anschaulichkeit).

Keine Kodierung, wenn nur eine Methode/Sozialform ohne Motivationsbezug genannt wird (PK) oder wenn kein Bezug zu den vorher kritisierten Aspekten erkennbar ist.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Decision-Making-Aussage (kein Vorschlag) ODER
- Vorschlag ohne Bezug zu Motivation/Interesse
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- einzelne Alternative, wenig konkret, ohne Umsetzungsbeispiel
- z.B. „Alltagsbezug herstellen“, „Catch-Komponente einbauen“
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- mehrere Alternativen ODER
- konkrete Beispiele/Umsetzungsschritte, die erklären, wie Motivation/Interesse geweckt wird
""",
        },
        "examples": [
            {"text": "Die Lehrerin könnte eine Geschichte aus dem Alltag erzählen, die Kinder vor ein Problem stellen oder mit einem Experiment in das Thema starten", "rating": 2},
            {"text": "Alltagsbezug und Relevanz sichtbar machen; Catch-Komponente einbauen", "rating": 1},
            {"text": "Genauere Fragen stellen", "rating": 0},
        ],
    },
    "1_Dm_kA": {
        "name": "Unterrichtseinstieg - Beschreiben/Merken - kognitive Aktivierung",
        "definition": """Decision Making (Dm): Handlungsalternative zur Verbesserung der kognitiven Aktivierung im Unterrichtseinstieg.

Kodiert wird, ob die Antwort beschreibt, wie die Lehrkraft (aus fachdidaktischer Perspektive) anders handeln würde, um den Einstieg kognitiv aktivierender zu gestalten (Problemorientierung, kognitiver Konflikt, tiefergehende Wiederholung, Vorwissensaktivierung, Hypothesen/Fragestellungen).

Keine Kodierung: reine Methoden-/Sozialformvorschläge ohne inhaltlich-kognitive Aktivierung (PK) oder „Experiment“ ohne Erklärung, wie dadurch kognitive Aktivierung entsteht.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Decision-Making-Aussage ODER
- Vorschlag ohne Bezug zu kognitiver Aktivierung im Einstieg
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- einzelne Alternative, nicht konkret/ohne Umsetzungsbeispiel
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- mehrere Alternativen und/oder konkrete Beispiele (z.B. Problemfrage, Verfremdung, Hypothesen)
- erläutert, wie das die kognitive Aktivierung steigert
""",
        },
        "examples": [
            {"text": "Ich würde problemorientiert bzw. durch Verfremdung einsteigen: z.B. ein Bild von einem Menschen im Bikini im Schnee zeigen.", "rating": 2},
            {"text": "Zu Beginn evtl. kleines Quiz zur Wiederholung (Vorwissenaktivierung)", "rating": 1},
            {"text": "Eine andere Methode benutzen", "rating": 0},
        ],
    },
    "1_E_T_IV": {
        "name": "Unterrichtseinstieg - Einschätzen - Tiefe/Inhaltliche Vielfalt",
        "definition": """Explanation (E_T_IV): Theoriebezug zur Begründung, warum die Einstiegs-Situation verbesserungsfähig ist.

Kodiert wird, ob die Antwort die beobachteten Aspekte mit der professionellen Wissensbasis verknüpft (Theoriebezug). Ein Theoriebezug liegt vor, wenn einschlägige Schlagwörter genannt oder eindeutig umschrieben werden.

Beispiele für relevante Schlagwörter (u.a.):
- kognitive Aktivierung, Schüleraktivierung, Basiskonzeptorientierung (z.B. Struktur–Funktion), kognitiver Konflikt, Problemorientierung, Vorwissensaktivierung
- Interessentheorie, situationales Interesse, Catch-Komponente, sachbezogener Anreiz, momentane Lernmotivation/Lernbereitschaft, Diskrepanzerlebnisse

Hinweis: Floskeln oder generische Kommentare ohne Theoriebezug sind niedriger zu kodieren.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Explanation_Theorie vorhanden
""",
            1: """Floskel / generischer Kommentar ohne Theorienennung
- alltagssprachliche Erklärung ohne erkennbaren Theoriebezug
- auch „sollte“-Aussagen ohne Theoriebezug
""",
            2: """Nennung/umschreibbarer Theoriebezug
- einschlägiges Konzept/Schlagwort wird genannt oder eindeutig umschrieben
- kann auch als „sollte“-Aussage auftreten
""",
            3: """Theoriebezug + zweckmäßige Begründung
- Theorie/Konzept wird genannt und die Notwendigkeit/Zweck wird begründet (Bezug zum Beispiel)
""",
        },
        "examples": [
            {"text": "Catch-Komponente kann situationales Interesse fördern, SuS sind mehr interessiert an dem Thema", "rating": 3},
            {"text": "LK fragt nur was letzte Stunde besprochen wurde ohne aktivierenden Kontext (kein Catch)", "rating": 2},
            {"text": "Begrüßung hat gefehlt", "rating": 0},
        ],
    },
    
    # Topic 3: Fachsprache (Technical Language)
    "3_D_F": {
        "name": "Fachsprache - Beschreiben - Fachsprache",
        "definition": """Beschreiben (Description): Fehlende Erklärung/Einbettung von Fachbegriffen (Funktion/Definition/Verknüpfung).

Kodiert wird, ob die Antwort beschreibt, dass Fachbegriffe genannt/verwendet werden, ohne dass sie definiert, erklärt oder funktional in einen Kontext eingeordnet werden (z.B. Hornschicht, Lederhaut, Unterhaut).

Hinweis: Diese Kategorie zielt auf die fehlende Erklärung/Verknüpfung ab (nicht primär auf Menge oder Schwierigkeit der Sprache).""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zur fehlenden Erklärung/Verknüpfung von Fachbegriffen
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- eher vage (z.B. „Fachbegriffe erklären“) ohne Details
""",
            2: """Systematisches / vollständiges Beschreiben
- konkret: welche Begriffe/Funktionen fehlen, wie zeigt sich das im Unterricht (sichtbar im Video)
""",
        },
        "examples": [
            {"text": "Fachbegriffe werden nicht erklärt", "rating": 2},
            {"text": "keine Klärung der Fachbegriffe", "rating": 1},
            {"text": "Overhead-Projektor", "rating": 0},
        ],
    },
    "3_D_Qual": {
        "name": "Fachsprache - Beschreiben - Qualität",
        "definition": """Beschreiben (Description): Qualität/Angemessenheit der Sprache (Wissenschaftssprache vs. Unterrichtssprache).

Kodiert wird, ob die Antwort beschreibt, dass die Lehrkraft zu anspruchsvolle/zu wissenschaftliche Sprache verwendet, die für SuS schwer verständlich ist (z.B. „taktile Wahrnehmung“), bzw. die Sprache nicht an die Lernenden angepasst ist.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zur (Un-)Angemessenheit der Sprache
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- vage Hinweise (z.B. „Fachsprache“) ohne nähere Ausführung
""",
            2: """Systematisches / vollständiges Beschreiben
- konkret: wissenschaftliche/komplizierte Sprache, SuS-Verständnisprobleme, Beispiele aus dem Unterricht
""",
        },
        "examples": [
            {"text": "Fachbegriffe werden eingeführt ohne Erklärung (z.B. Vesikel)", "rating": 2},
            {"text": "Fachsprache-Alltagssprache", "rating": 1},
            {"text": "Längere Zeit zum Anschauen", "rating": 0},
        ],
    },
    "3_D_Quan": {
        "name": "Fachsprache - Beschreiben - Quantität",
        "definition": """Beschreiben (Description): Quantität/Menge an Fachbegriffen.

Kodiert wird, ob die Antwort beschreibt, dass zu viele Fachbegriffe/Details/Informationen auf einmal eingeführt werden (Begriffsflut; „zu viel Input“), wodurch Überforderung begünstigt wird.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zur Menge an Fachbegriffen
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- vage („zu viel auf einmal“) ohne weitere Spezifizierung
""",
            2: """Systematisches / vollständiges Beschreiben
- klarer Bezug zur Begriffs-/Infoflut, ggf. mit Beispielen (mehrere Begriffe/Tempo/Überforderung)
""",
        },
        "examples": [
            {"text": "viele Fachbegriffe (Vesikel, Melanin, UV-Strahlen...)", "rating": 2},
            {"text": "Alles auf einmal", "rating": 1},
            {"text": "Beschreiben des Bildes", "rating": 0},
        ],
    },
    "3_Dm_F": {
        "name": "Fachsprache - Decision Making - Fachsprache",
        "definition": """Decision Making (Dm): Handlungsalternative zur Verbesserung der Erklärung/Verknüpfung von Fachbegriffen.

Kodiert wird, ob die Antwort beschreibt, wie die Lehrkraft Fachbegriffe definieren/erklären bzw. (optional) mit Funktionen/Bezügen verknüpfen lassen würde (z.B. im Gespräch mit der Klasse).""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Vorschlag zur Erklärung/Definition/Verknüpfung von Fachbegriffen
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- einzelne Alternative, wenig konkret
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- konkrete Umsetzung/Beispiele (z.B. Begriffe an die Tafel + Definition; Funktionen gemeinsam erarbeiten)
""",
        },
        "examples": [
            {"text": "Klaren Arbeitsauftrag geben, klare Fragen stellen, Erläuterungen der Fachbegriffe beigeben und durch geeignete Modelle unterstützen.", "rating": 2},
            {"text": "Fachbegriffe an die Tafel schreiben mit Definition, die für die SuS verständlich ist", "rating": 1},
            {"text": "Mehr Arbeitsblätter für Selbstständigkeit", "rating": 0},
        ],
    },
    "3_Dm_Kons": {
        "name": "Fachsprache - Decision Making - Konsistenz",
        "definition": """Decision Making (Dm): Konstruktive (kognitiv aktive) Auseinandersetzung mit Fachbegriffen ermöglichen.

Kodiert wird, ob die Antwort Vorschläge macht, bei denen SuS Fachbegriffe selbstständig/aktiv (konstruktiv) bearbeiten, statt nur reproduktiv wiederzugeben (z.B. Zuordnungsaufgaben, eigenaktive Erarbeitung, Überprüfen richtiger/falscher Aussagen).

Definition „konstruktiv“ (Kurzfassung): aktive kognitive Auseinandersetzung; reine Reproduktion reicht nicht aus.""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Vorschlag zur konstruktiven SuS-Auseinandersetzung mit Fachbegriffen
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- SuS sollen „mehr einbezogen werden“ ohne konkrete kognitive Aufgabe
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- konkrete Aufgaben/Settings, in denen SuS Fachbegriffe aktiv zuordnen/erarbeiten/prüfen
""",
        },
        "examples": [
            {"text": "SuS beim Erstellen des Tafelbilds mit einbeziehen: SuS legen die Karten/Bestandteile selbst an die Tafel an.", "rating": 2},
            {"text": "Schüler selbst die Bestandteile an die Tafel kleben lassen", "rating": 1},
            {"text": "Film zeigen", "rating": 0},
        ],
    },
    "3_Dm_Qual": {
        "name": "Fachsprache - Decision Making - Qualität",
        "definition": """Decision Making (Dm): Sprache an SuS anpassen (Unterrichtssprache statt Wissenschaftssprache).

Kodiert wird, ob die Antwort beschreibt, wie die Lehrkraft sprachlich verständlicher formulieren würde (z.B. Alltags-/Unterrichtssprache nutzen, Fachsprache übersetzen/erklären), damit SuS die Inhalte zugänglich verstehen.""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Vorschlag zur sprachlichen Angemessenheit
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- z.B. „mehr Alltagssprache“ ohne konkrete Umsetzung
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- konkrete Beispiele (Begriff erklären/übersetzen, passende Unterrichtssprache wählen)
""",
        },
        "examples": [
            {"text": "Mehrere SuS antworten lassen; bei niedrigen Klassenstufen auf schweres Fachvokabular verzichten", "rating": 2},
            {"text": "Die Lehrkraft sollte mehr in Alltagssprache reden und weniger Fachbegriffe zum Erklären verwenden", "rating": 1},
            {"text": "Lehrkraft macht ein Tafelbild welches die SuS abschreiben sollen", "rating": 0},
        ],
    },
    "3_Dm_Quan": {
        "name": "Fachsprache - Decision Making - Quantität",
        "definition": """Decision Making (Dm): Quantität der Fachbegriffe reduzieren.

Kodiert wird, ob die Antwort beschreibt, wie die Lehrkraft weniger/ausgewählte Fachbegriffe einführen würde (didaktische Reduktion), um Überforderung durch Begriffsflut zu vermeiden.""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Vorschlag zur Reduktion/Strukturierung der Fachbegriffsmenge
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- z.B. „weniger Begriffe“ ohne weitere Konkretisierung
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- konkrete Auswahl/Begründung oder Kombination mit erklärender Einführung
""",
        },
        "examples": [
            {"text": "Weniger Fachbegriffe und die wirklich erklären", "rating": 2},
            {"text": "Weniger Fachbegriffe verwenden", "rating": 1},
            {"text": "SUS mehr ins Geschehniss einbeziehen", "rating": 0},
        ],
    },
    "3_E_T_IV": {
        "name": "Fachsprache - Explanation - Tiefe/Inhaltliche Vielfalt",
        "definition": """Explanation (E_T_IV): Theoriebezug zur Begründung von Problemen mit Fachsprache.

Kodiert wird, ob die Antwort die Beobachtungen mit Theorie/Schlagwörtern verknüpft, z.B.:
- Begriffsflut/Überforderung (Arbeitsgedächtnis, Cognitive Load)
- passende Unterrichtssprache vs. Wissenschaftssprache
- Kontextualisierung/Einordnung von Fachbegriffen

Floskeln ohne Theoriebezug sind niedriger zu kodieren.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Explanation_Theorie
""",
            1: """Floskel / generischer Kommentar ohne Theorienennung
""",
            2: """Nennung/umschreibbarer Theoriebezug
- z.B. Cognitive overload, Arbeitsgedächtnis, Unterrichtssprache
""",
            3: """Theoriebezug + zweckmäßige Begründung
- Theorie wird genannt und mit Zweck/Notwendigkeit begründet
""",
        },
        "examples": [
            {"text": "Es wurden zu viele den Schülern unbekannte Fachbegriffe genutzt. Dies überfordert die SuS. (Stichwort: Cognitive overload)", "rating": 2},
            {"text": "Arbeitsspeicher des Gedächtnisses", "rating": 1},
            {"text": "Neues Wissen wird nicht notiert.", "rating": 0},
        ],
    },
    
    # Topic 4: Experimente (Experiments/Scientific Inquiry)
    "4_D_Ep": {
        "name": "Experimente - Beschreiben - Erkenntnisgewinnung/Prozesse",
        "definition": """Beschreiben (Description): Fehlende Schritte im Erkenntnisprozess beim Experimentieren (Scientific Inquiry).

Kodiert wird, ob die Antwort beschreibt, dass zentrale Erkenntnisschritte fehlen oder nicht gemeinsam mit SuS erarbeitet werden, z.B.:
- fehlende Fragestellung (Sinn/Zweck)
- fehlende Hypothesenbildung
- Experimentplanung ohne SuS-Beteiligung
- Auswertung/Interpretation unzureichend (zu wenige Daten/Schülerbeiträge)
- „rezeptartig“/eher Versuch als Experiment, Ziel unklar
""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zum Erkenntnisprozess
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- vage (z.B. „keine Fragestellung“) ohne weitere Einbettung
""",
            2: """Systematisches / vollständiges Beschreiben
- konkrete Benennung mehrerer fehlender Schritte/Details (Hypothese, Auswertung, Beteiligung)
""",
        },
        "examples": [
            {"text": "Hypothesen generieren fehlt", "rating": 2},
            {"text": "Eigentlich ein Versuch, kein Experiment", "rating": 1},
            {"text": "Klingel", "rating": 0},
        ],
    },
    "4_Dm_Ep": {
        "name": "Experimente - Beschreiben/Merken - Erkenntnisgewinnung/Prozesse",
        "definition": """Decision Making (Dm): Handlungsalternative zur Förderung des Erkenntnisprozesses beim Experimentieren.

Kodiert wird, ob die Antwort Vorschläge macht, bei denen SuS einzelne Erkenntnisschritte selbst vollziehen (z.B. Frage -> Hypothese -> Planung/Durchführung -> Datenanalyse/Interpretation).

Kodiert wird, ob die Antwort Vorschläge macht, bei denen SuS einzelne Erkenntnisschritte selbst vollziehen (z.B. Frage → Hypothese → Planung/Durchführung → Datenanalyse/Interpretation).

Keine Kodierung: reine Aussagen zur Experimentart (Schülerexperiment vs. Lehrerexperiment) ohne Bezug zu Erkenntnisschritten.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Decision-Making-Aussage ODER ohne Bezug zum Erkenntnisprozess
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- einzelne Alternative/Schritt genannt, wenig konkret
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- mehrere Schritte/konkrete Beispiele; erläutert, wie die Alternative das Kritisierte verbessert
""",
        },
        "examples": [
            {"text": "Experiment mit dem naturwissenschaftlichen Weg der Erkenntnisgewinnung: 1. Frage formulieren 2. Hypothese generieren 3. Untersuchungen planen 4. Daten analysieren", "rating": 2},
            {"text": "SuS alle experimentieren lassen und davor sollen sie Vermutungen über das Experiment aufstellen", "rating": 1},
            {"text": "Schulranzen wegräumen", "rating": 0},
        ],
    },
    "4_E_T_IV": {
        "name": "Experimente - Einschätzen - Tiefe/Inhaltliche Vielfalt",
        "definition": """Explanation (E_T_IV): Theoriebezug zur Begründung von Problemen im Erkenntnisprozess/Experimentieren.

Kodiert wird, ob die Antwort die Beobachtungen mit Theorie/Schlagwörtern verknüpft, z.B.:
- Erkenntnisgewinnung / Kompetenzbereich Erkenntnisgewinnung
- Erkenntnisprozess, wissenschaftliches Denken (scientific reasoning), wissenschaftliche Arbeitsweise
- hypothesen-geleitetes Experimentieren, hypothetisch-deduktiver Erkenntnisweg
- Bildungsstandards
""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Explanation_Theorie
""",
            1: """Floskel / generischer Kommentar ohne Theorienennung
""",
            2: """Nennung/umschreibbarer Theoriebezug
- z.B. Erkenntnisgewinnung, wissenschaftliche Arbeitsweise, Hypothesenbildung
""",
            3: """Theoriebezug + zweckmäßige Begründung
- Theorie wird genannt und Zweck/Notwendigkeit wird begründet
""",
        },
        "examples": [
            {"text": "SuS üben nicht die naturwissenschaftliche Arbeitsweise ein", "rating": 1},
            {"text": "Erkenntnisgewinnung: Frage -> Hypothese -> Planung/Durchführung -> Analysieren/Interpretieren", "rating": 2},
            {"text": "Schüler-Lehrer Beziehung", "rating": 0},
        ],
    },
    
    # Topic 5: Modelle (Models)
    "5_D_Ma": {
        "name": "Modelle - Beschreiben - Modellanwendung",
        "definition": """Beschreiben (Description): Niveau der Modellarbeit / Modellanwendung.

Kodiert wird, ob die Antwort beschreibt, dass das Modell nur oberflächlich genutzt wird (z.B. nur Teile benennen lassen) und nicht zweckmäßig eingebettet bzw. zum Erklären von Zusammenhängen genutzt wird.

Beispiele:
- Modell dient nur zur Veranschaulichung/Beschreibung (Niveau I)
- Modell wird unvollständig beschrieben; Zweck/Einführung unklar
- funktionale Beziehungen/Zusammenhänge werden nicht mit Hilfe des Modells erarbeitet
""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zur Modellanwendung
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- vage („Einsatz des Modells“) ohne weitere Spezifizierung
""",
            2: """Systematisches / vollständiges Beschreiben
- konkrete Hinweise, wie das Modell genutzt wird und warum das unzureichend ist
""",
        },
        "examples": [
            {"text": "Modell wird nicht nochmal ganz erklärt", "rating": 2},
            {"text": "Einsatz des Modells", "rating": 1},
            {"text": "Farben", "rating": 0},
        ],
    },
    "5_D_Mk": {
        "name": "Modelle - Beschreiben - Modellkompetenz",
        "definition": """Beschreiben (Description): Fehlende Modellkritik / Modellkompetenz.

Kodiert wird, ob die Antwort beschreibt, dass keine (oder unzureichende) Modellkritik stattfindet, d.h. Unterschiede zwischen Modell und Original, Entsprechungen, Grenzen oder Beiwerk nicht thematisiert werden.

Hinweis aus dem Manual: Wenn nur „fehlende Modellkritik“ genannt wird, kodiert man typischerweise mit 1. Wenn weitere beschreibende Aspekte hinzukommen (z.B. Vergleich Original–Modell, Farben/Beiwerk, mehrere Modelle), kann 2 vergeben werden.""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zur Modellkritik/Modellkompetenz
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- „keine Modellkritik“ o.ä. ohne weitere Details
""",
            2: """Systematisches / vollständiges Beschreiben
- zusätzliche Details/Aspekte (z.B. Grenzen, Beiwerk, Vergleich zum Original)
""",
        },
        "examples": [
            {"text": "Es gab keine Modellkritik", "rating": 1},
            {"text": "Unterschiede zwischen original und Modell darstellen", "rating": 2},
            {"text": "Wiederholung", "rating": 0},
        ],
    },
    "5_Dm_Ma": {
        "name": "Modelle - Beschreiben/Merken - Modellanwendung",
        "definition": """Decision Making (Dm): Handlungsalternative zur verbesserten Modellanwendung.

Kodiert wird, ob die Antwort beschreibt, wie das Modell zweckmäßiger genutzt werden kann (mindestens Niveau II: Zusammenhänge erklären; ggf. Niveau III: Hypothesen/Voraussagen am Modell) und wie der Zweck/Einführung des Modells deutlich gemacht wird.""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Vorschlag zur Modellanwendung
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- einzelne Alternative, wenig konkret
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- mehrere Schritte/konkrete Beispiele (Zweck klären, Zusammenhänge am Modell, Hypothesen)
""",
        },
        "examples": [
            {"text": "Modell und Bild des Original gleichzeitig zeigen und SuS die Strukturen im Model und Original zuordnen lassen", "rating": 2},
            {"text": "Erstmal auf das Modell eingehen und auf die Größe eingehen", "rating": 1},
            {"text": "s. vorher", "rating": 0},
        ],
    },
    "5_Dm_Mk": {
        "name": "Modelle - Beschreiben/Merken - Modellkompetenz",
        "definition": """Decision Making (Dm): Handlungsalternative zur Modellkritik / Förderung der Modellkompetenz.

Kodiert wird, ob die Antwort beschreibt, wie Modellkritik eingebaut werden soll (Entsprechungen, Grenzen, Beiwerk; Vergleich zum Original; Vor-/Nachteile; ggf. mehrere Modelle), um Fehlvorstellungen zu vermeiden.""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Vorschlag zur Modellkritik/Modellkompetenz
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- z.B. „Modellkritik durchführen“ ohne Details
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- nennt konkrete Kritikpunkte/Aspekte (Grenzen/Beiwerk/Entsprechungen, Vergleich Original–Modell)
""",
        },
        "examples": [
            {"text": "Modellkritik mit den Schülern besprechen: Leistungen, Grenzen, Vergleich mit der Realität", "rating": 2},
            {"text": "Man sollte den Kindern bewusst machen, dass das Modell so nicht der Realität entspricht", "rating": 1},
            {"text": "Nicht verbesserungsfähig", "rating": 0},
        ],
    },
    "5_E_T_IV": {
        "name": "Modelle - Einschätzen - Tiefe/Inhaltliche Vielfalt",
        "definition": """Explanation (E_T_IV): Theoriebezug zur Begründung von Problemen bei der Arbeit mit Modellen.

Kodiert wird, ob die Antwort einen Theoriebezug herstellt, z.B.:
- Zweck/Einführung von Modellen (Modellbildung)
- Niveaustufen der Modellarbeit (z.B. Veranschaulichung vs. Erklären von Zusammenhängen vs. Hypothesen/Voraussagen)
- Modellkritik (Entsprechungen, Grenzen, Beiwerk) zur Vermeidung von Fehlvorstellungen
""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Explanation_Theorie
""",
            1: """Floskel / generischer Kommentar ohne Theorienennung
""",
            2: """Nennung/umschreibbarer Theoriebezug
- z.B. Modellkritik, Fehlvorstellungen, Zweck von Modellen
""",
            3: """Theoriebezug + zweckmäßige Begründung
- Theorie wird genannt und mit Zweck/Notwendigkeit begründet
""",
        },
        "examples": [
            {"text": "Modelle dienen als Denk- und Arbeitsweise zur Erkenntnisgewinnung; hier dient es nur als Anschauungsobjekt.", "rating": 3},
            {"text": "Modellkritik ist zentraler Aspekt; ohne Modellkritik können Fehlvorstellungen entstehen.", "rating": 2},
            {"text": "Modell ist ziemlich klein", "rating": 0},
        ],
    },
    
    # Topic 6: Sicherung/Transfer (Consolidation/Transfer)
    "6_D_Rb": {
        "name": "Sicherung/Transfer - Beschreiben - Reflexion/Begründung",
        "definition": """Beschreiben (Description): Rückbezug/Reflexion/Begründung (Rb) in der Sicherungs-/Transferphase.

Kodiert wird, ob die Antwort beschreibt, dass kein (oder unzureichender) Rückbezug zum Einstieg/zu einer Problem- bzw. Fokusfrage stattfindet (z.B. Einstiegsgeschichte wird nicht wieder aufgegriffen).""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zum Rückbezug/Reflexion
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- vage („Rückbezug fehlt“) ohne weitere Details
""",
            2: """Systematisches / vollständiges Beschreiben
- klarer Bezug: Einstieg/Problemfrage wird nicht aufgegriffen; konkrete Hinweise aus der Szene
""",
        },
        "examples": [
            {"text": "Rückbezug zum Einstieg findet nicht statt", "rating": 2},
            {"text": "Rückbezug zur Fokusfrage", "rating": 1},
            {"text": "Präsentation", "rating": 0},
        ],
    },
    "6_D_kA": {
        "name": "Sicherung/Transfer - Beschreiben - kognitive Aktivierung",
        "definition": """Beschreiben (Description): Kognitive Aktivierung in der Sicherungs-/Transferphase.

Kodiert wird, ob die Antwort beschreibt, dass die Sicherung/der Abschluss überwiegend reproduktiv ist und keine stärkeren kognitiven Prozesse aktiviert werden (z.B. keine Vernetzung, kein Transfer, keine Struktur–Funktion-Verknüpfung).""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Description ODER kein Bezug zu kognitiver Aktivierung in der Sicherung
""",
            1: """Unsystematisches / unvollständiges Beschreiben
- vage („nur Wiederholung“, „kein Transfer“) ohne weitere Details
""",
            2: """Systematisches / vollständiges Beschreiben
- klare Beschreibung reproduktiver Sicherung; konkrete Hinweise (Tafelbild ablesen, keine Vertiefung)
""",
        },
        "examples": [
            {"text": "Kein Transfer, reine Reproduktion", "rating": 2},
            {"text": "Faktenwissen statt Zusammenhänge von Struktur und Funktion", "rating": 1},
            {"text": "Aufmerksamkeit der SuS", "rating": 0},
        ],
    },
    "6_Dm_Rb": {
        "name": "Sicherung/Transfer - Beschreiben/Merken - Reflexion/Begründung",
        "definition": """Decision Making (Dm): Handlungsalternative zum Rückbezug/Reflexion am Stundenende.

Kodiert wird, ob die Antwort beschreibt, wie ein Rückbezug zur Problem-/Fokusfrage bzw. zum Einstieg hergestellt wird (z.B. Gegenüberstellung Einstieg–Ergebnis; erneuter Bezug zur Einstiegsgeschichte).""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Rückbezug-Vorschlag
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- z.B. „Rückbezug auf Fokusfrage“ ohne Ausführung
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- konkrete Fragen/Beispiele, wie der Rückbezug gestaltet wird
""",
        },
        "examples": [
            {"text": "Rückbezug zur Ritterrüstung: Unterschiede und warum, und damit abschließen", "rating": 2},
            {"text": "Rückbezug auf die Fokusfrage zum Anfang der Stunde", "rating": 1},
            {"text": "Eine Präsentation mit mehreren Bildern verwenden", "rating": 0},
        ],
    },
    "6_Dm_kA": {
        "name": "Sicherung/Transfer - Beschreiben/Merken - kognitive Aktivierung",
        "definition": """Decision Making (Dm): Handlungsalternative zur kognitiven Aktivierung in der Sicherungs-/Transferphase.

Kodiert wird, ob die Antwort beschreibt, wie der Abschluss kognitiv aktivierender gestaltet werden soll (Vernetzung, Struktur–Funktion-Verknüpfung, Transferaufgaben, aktive Aufgaben statt Ablesen).""",
        "rating_scale": {
            0: """trifft nicht zu
- kein Vorschlag zur kognitiven Aktivierung am Ende
""",
            1: """Handlungsalternative mit Bezug, aber eher allgemein
- z.B. „Transferaufgabe einbauen“ ohne Ausführung
""",
            2: """Handlungsalternative mit Bezug, konkret/ausführlich
- konkrete Aufgaben/Beispiele (Tafel abdecken, Zuordnen, Transfer auf neue Situationen)
""",
        },
        "examples": [
            {"text": "Tafel abdecken und Kinder die Schichten/Funktionen erneut zuordnen lassen, danach Transferaufgabe", "rating": 2},
            {"text": "Anstatt des Wiederholens eine Transferaufgabe stellen", "rating": 1},
            {"text": "Ich würde es genauso machen", "rating": 0},
        ],
    },
    "6_E_T_IV": {
        "name": "Sicherung/Transfer - Einschätzen - Tiefe/Inhaltliche Vielfalt",
        "definition": """Explanation (E_T_IV): Theoriebezug zur Begründung von Rückbezug/Transfer bzw. kognitiver Aktivierung am Stundenende.

Kodiert wird, ob die Antwort Theorie/Schlagwörter zur Begründung nutzt, z.B.:
- problemorientiert, Problemfrage/Fokusfrage, Rückbezug, Erkenntnisprozess
- Transfer (Ausweitung/Übertragung), Vernetzung, Reorganisation
- Basiskonzept Struktur und Funktion (Vernetzung/Verarbeitungstiefe)
""",
        "rating_scale": {
            0: """trifft nicht zu
- keine Explanation_Theorie
""",
            1: """Floskel / generischer Kommentar ohne Theorienennung
""",
            2: """Nennung/umschreibbarer Theoriebezug
- z.B. Transfer, Basiskonzept, Fokusfrage, Vernetzung
""",
            3: """Theoriebezug + zweckmäßige Begründung
- Theorie wird genannt und der Zweck wird begründet
""",
        },
        "examples": [
            {"text": "Die Sicherung/Transferphase fand nur als reine Wiederholung statt; ohne Basiskonzeptorientierung wird nicht transferiert.", "rating": 3},
            {"text": "In der Vertiefungsphase soll das Gelernte in einem Transfer vertieft und vernetzt werden.", "rating": 2},
            {"text": "Es wird das bereits notierte wiederholt.", "rating": 0},
        ],
    },
}


def get_prompt(category: str, text: str, template_name: str = "baseline", output_format: str = "text") -> str:
    """Get formatted prompt for a specific category.

    The function prefers a category-specific `system_prompt` if present in the category
    definition; otherwise it generates a concise category-level instruction that
    enforces strict output formatting.
    
    Args:
        category: DiKoBi category (e.g., '1_D_M')
        text: Text to be rated
        template_name: Prompt template to use (baseline, few_shot, etc.)
        output_format: Output format - "json" or "text" (default: "text")
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}")

    cat = CATEGORIES[category]

    # All categories should have the standard structure
    if "definition" not in cat or "rating_scale" not in cat:
        raise ValueError(f"Category {category} is missing required fields (definition or rating_scale)")

    # Determine valid output range
    valid_range = list(cat["rating_scale"].keys())
    valid_range_str = ", ".join(str(x) for x in sorted(valid_range))

    # Use category-specific system prompt if provided, otherwise use global default
    use_json = (output_format == "json")
    if use_json:
        cat_system = cat.get("system_prompt_json", get_system_prompt(use_json=True))
    else:
        cat_system = cat.get("system_prompt", get_system_prompt(use_json=False))

    # Build rating scale text
    scale_text = "\n\n".join([
        f"{score}: {description}"
        for score, description in sorted(cat["rating_scale"].items())
    ])

    # Base prompt structure
    base_prompt = f"""{cat_system}

Category: {cat['name']}

{cat['definition']}

Rating Scale:
{scale_text}"""

    # Add few-shot examples if requested and available for this category
    examples_text = ""
    if "examples" in cat:
        # Preferred mode: include ALL curated examples
        if template_name in ["few_shot", "few_shot_all"]:
            examples = cat["examples"]
        # Legacy modes: include first N examples (kept for backward compatibility)
        elif template_name in ["few_shot_1", "few_shot_2", "few_shot_3"]:
            num_examples = int(template_name.split("_")[-1])  # Extract number (1, 2, or 3)
            examples = cat["examples"][:num_examples]
        else:
            examples = []
        
        if examples:
            examples_text = "\n\nExamples:\n\n"
            for i, example in enumerate(examples, 1):
                examples_text += f"""Example {i}:
Text: \"{example['text']}\"
Rating: {example['rating']}

"""
    
    # Construct final prompt
    if use_json:
        # JSON format ending
        prompt = f"""{base_prompt}{examples_text}

Text to rate:
\"{text}\"

Valid ratings: {valid_range_str}

Respond with valid JSON only:"""
    else:
        # Text format ending
        prompt = f"""{base_prompt}{examples_text}

Text to rate:
\"{text}\"

Valid ratings: {valid_range_str}
Rating:"""  # Ends with "Rating:" to prompt just the number
    
    return prompt


def _get_valid_outputs_from_standards(category: str) -> list:
    """
    Derive valid outputs from MAX_VALUES in standards.py.
    
    Args:
        category: Category name (e.g., '1_D_M')
        
    Returns:
        List of valid output integers [0, 1, ..., max_value]
        
    Raises:
        ValueError: If category not found in standards
    """
    if category not in MAX_VALUES:
        raise ValueError(f"Category {category} not found in standards.MAX_VALUES")
    
    max_val = MAX_VALUES[category]
    
    # Handle infinite max (e.g., for count variables)
    if max_val == float('inf'):
        # For count variables, we can't enumerate all possibilities
        # Return empty list to indicate unlimited range
        return []
    
    # Return [0, 1, 2, ..., max_val]
    return list(range(int(max_val) + 1))


def get_valid_outputs(category: str) -> list:
    """
    Get valid output range for a category from standards.py.
    
    Args:
        category: Category name
        
    Returns:
        List of valid output integers
        
    Raises:
        ValueError: If category not found in standards
    """
    return _get_valid_outputs_from_standards(category)


def list_categories() -> list:
    """Get list of all available categories."""
    return list(CATEGORIES.keys())


def list_prompt_templates() -> dict:
    """
    Get list of available prompt templates.
    
    Returns:
        Dictionary mapping display names to template keys.
        
    Available templates:
        - zero_shot: No examples, just category definition and rating scale
        - few_shot: Include all curated examples for the category

    Legacy templates (still accepted by get_prompt for backward compatibility):
        - few_shot_1 / few_shot_2 / few_shot_3: Include the first N examples
    """
    return {
        "zero_shot": "zero_shot",
        "few_shot": "few_shot",
    }
