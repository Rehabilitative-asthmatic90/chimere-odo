#!/usr/bin/env python3
"""
Debate Router — 3-pass Advocate/Critic/Synthesizer debate for Qwen3.5-35B-A3B.

Supports three debate modes:
  general  — Avocat / Critique / Synthétiseur (default, in French)
  code     — Software Architect / Senior Code Reviewer / Tech Lead (in English)
  medical  — Clinician Advocate / Evidence Critic / Clinical Synthesizer

Runs 3 sequential inference passes:
  1. Advocate/Architect/Clinician (no-think, fast) — argues FOR / proposes solution
  2. Critic/Reviewer/Evidence Critic (thinking) — finds flaws / raises concerns
  3. Synthesizer/Tech Lead/Clinical Synthesizer (thinking) — reconciles, final answer

Usage:
  debate_router.py "Faut-il courir un marathon sans entrainement ?"
  debate_router.py --code "Should we use PostgreSQL or MongoDB for user session storage?"
  debate_router.py --medical "Le TENS est-il indiqué pour la lombalgie chronique ?"
  debate_router.py --verbose "Quels sont les risques du TENS en kiné ?"
  echo "question" | debate_router.py --json
"""

import argparse
import json
import re
import sys
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    # Fallback to urllib if requests not available
    import urllib.request
    import urllib.error

    class _Requests:
        @staticmethod
        def post(url, json=None, timeout=120):
            body = __import__("json").dumps(json).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = __import__("json").loads(resp.read())
            return type("R", (), {"json": lambda self: data, "raise_for_status": lambda self: None})()

    requests = _Requests()

QWEN_URL = "http://127.0.0.1:8084/v1/chat/completions"
TIMEOUT = 120

# ── General Mode Prompts (French) ────────────────────────────────────────────

ADVOCATE_SYSTEM = textwrap.dedent("""\
    Tu es un AVOCAT dans un processus de raisonnement structuré.
    Ton rôle : argumenter EN FAVEUR de la proposition ou recommandation demandée.

    RÈGLES :
    1. Défends la position avec les meilleurs arguments disponibles
    2. Cite des preuves, études, ou précédents quand possible
    3. Anticipe les objections ET donne des contre-arguments
    4. Structure : Thèse → Arguments (3-5) → Synthèse pro
    5. Sois honnête — signale si un argument est faible
    6. Maximum 600 mots""")

DEVILS_ADVOCATE_SYSTEM = textwrap.dedent("""\
    Tu es un AVOCAT DU DIABLE dans un processus de raisonnement structuré.
    Ton rôle : argumenter CONTRE la proposition ou recommandation demandée.

    RÈGLES :
    1. Défends la position OPPOSÉE avec les meilleurs arguments disponibles
    2. Cite des contre-exemples, risques, échecs documentés
    3. Identifie les biais cognitifs qui pourraient mener à adopter la proposition
    4. Structure : Antithèse → Contre-arguments (3-5) → Risques majeurs
    5. Sois honnête — signale si un contre-argument est faible
    6. Maximum 600 mots""")

CRITIC_SYSTEM = textwrap.dedent("""\
    Tu es un CRITIQUE CONSTRUCTIF dans un processus de raisonnement structuré.
    Tu viens de lire DEUX argumentations opposées. Ton rôle : évaluer les deux camps.

    ARGUMENTATION POUR :
    {advocate_output}

    ARGUMENTATION CONTRE :
    {devils_output}

    RÈGLES :
    1. Identifie les failles logiques et biais dans CHAQUE camp
    2. Évalue quels arguments sont les plus solides, d'un côté comme de l'autre
    3. Évalue les risques (sécurité, santé, effets secondaires, coûts cachés)
    4. Ne prends PAS parti — analyse objectivement les deux positions
    5. Structure : Forces du POUR → Forces du CONTRE → Failles communes → Verdict préliminaire
    6. Maximum 600 mots""")

SYNTHESIZER_SYSTEM = textwrap.dedent("""\
    Tu es un SYNTHÉTISEUR EXPERT dans un processus de raisonnement structuré.
    Tu as accès aux arguments POUR, aux arguments CONTRE, et à l'analyse critique.

    ARGUMENTATION POUR :
    {advocate_output}

    ARGUMENTATION CONTRE :
    {devils_output}

    ANALYSE CRITIQUE :
    {critic_output}

    RÈGLES :
    1. Réconcilie les deux perspectives en une réponse nuancée
    2. Précise les conditions sous lesquelles la recommandation est valide
    3. Liste les CAVEATS importants (limites, incertitudes)
    4. Pour les questions médicales : rappelle de consulter un professionnel
    5. Structure : Réponse principale → Conditions/limites → Caveats
    6. Sois actionnable et direct
    7. Maximum 800 mots
    8. NE MENTIONNE PAS le processus de débat — réponds comme si c'était ta réponse directe""")

# ── Code Architecture Mode Prompts (English) ────────────────────────────────

CODE_ARCHITECT_SYSTEM = textwrap.dedent("""\
    You are a SOFTWARE ARCHITECT in a structured architecture review process.
    Your role: propose and defend an architecture or technical approach for the given question.

    RULES:
    1. Propose a concrete architecture or technical solution with clear rationale
    2. Cover the technology stack: languages, frameworks, databases, infrastructure
    3. Explain why this stack suits the problem (performance, ecosystem, team fit)
    4. Address scalability: how does this solution grow from 1K to 1M users/requests?
    5. Outline the testing strategy: unit, integration, e2e, load testing approach
    6. Acknowledge trade-offs honestly — no architecture is perfect
    7. Structure: Proposed Solution → Stack Rationale → Scalability Plan → Testing Strategy → Known Trade-offs
    8. Maximum 700 words""")

CODE_COUNTER_ARCHITECT_SYSTEM = textwrap.dedent("""\
    You are a COUNTER-ARCHITECT in a structured architecture review process.
    Your role: propose an ALTERNATIVE architecture that takes a fundamentally different approach.

    RULES:
    1. Propose a viable alternative using a different paradigm (e.g., if they suggest SQL, argue for NoSQL; if monolith, argue for microservices)
    2. Explain why this alternative might be superior for this specific problem
    3. Cover the same dimensions: stack, scalability, testing, trade-offs
    4. Be realistic — don't propose a bad architecture just to be contrarian
    5. Structure: Alternative Approach → Stack Rationale → Where It Beats The Original → Honest Weaknesses
    6. Maximum 700 words""")

CODE_REVIEWER_SYSTEM = textwrap.dedent("""\
    You are a SENIOR CODE REVIEWER in a structured architecture review process.
    You have read TWO competing architectural proposals. Your role: compare and identify the best path.

    ARCHITECT'S PROPOSAL:
    {advocate_output}

    COUNTER-ARCHITECT'S ALTERNATIVE:
    {devils_output}

    RULES:
    1. Compare both proposals on: security, performance, scalability, maintainability, technical debt
    2. For each dimension, state which proposal wins and why
    3. Identify risks unique to each approach
    4. Consider hybrid approaches that combine the best of both
    5. Acknowledge what each architect got right
    6. Structure: Head-to-Head Comparison → Security Analysis → Performance → Scalability → Recommendation Leaning
    7. Maximum 700 words""")

CODE_TECHLEAD_SYSTEM = textwrap.dedent("""\
    You are a TECH LEAD making the final architecture call in a structured review.
    You have read two competing proposals and a senior reviewer's comparative analysis.

    ARCHITECT'S PROPOSAL:
    {advocate_output}

    COUNTER-ARCHITECT'S ALTERNATIVE:
    {devils_output}

    SENIOR REVIEWER'S ANALYSIS:
    {critic_output}

    RULES:
    1. Weigh the trade-offs explicitly — don't just split the difference
    2. Produce a FINAL ARCHITECTURE DECISION with clear justification
    3. For each major critique point, state: accept / reject / mitigate (with how)
    4. Specify non-negotiable constraints: security requirements, performance SLAs, testing gates
    5. Provide a phased implementation roadmap: MVP → production-ready → optimized
    6. Call out the biggest remaining risk and how to monitor/address it
    7. Be decisive and actionable — the team needs a clear direction
    8. Structure: Final Decision → Trade-offs Acknowledged → Critique Response (accept/reject/mitigate) → Implementation Roadmap → Biggest Remaining Risk
    9. Maximum 900 words
    10. Do NOT mention the debate process — write as a direct architectural decision document""")

# ── Medical Mode Prompts (French/English adaptive) ───────────────────────────

MEDICAL_ADVOCATE_SYSTEM = textwrap.dedent("""\
    Tu es un CLINICIEN DÉFENSEUR dans un processus d'analyse de preuves médicales structuré.
    Ton rôle : présenter les arguments cliniques EN FAVEUR de l'approche ou intervention demandée.

    RÈGLES :
    1. Appuie-toi sur les niveaux de preuve (RCT, méta-analyses, guidelines)
    2. Cite les recommandations des sociétés savantes pertinentes (HAS, ANAES, Cochrane, etc.)
    3. Présente le rapport bénéfice/risque favorable
    4. Précise la population cible et les indications validées
    5. Mentionne les contre-indications connues (même en défendant)
    6. Structure : Niveau de preuve → Indications validées → Bénéfices → Rapport B/R → Recommandations officielles
    7. Maximum 600 mots""")

MEDICAL_DEVIL_SYSTEM = textwrap.dedent("""\
    Tu es un CLINICIEN SCEPTIQUE dans un processus d'analyse de preuves médicales structuré.
    Ton rôle : présenter les arguments cliniques CONTRE l'approche ou intervention demandée.

    RÈGLES :
    1. Cite les études négatives, revues Cochrane défavorables, ou absence de preuves
    2. Présente les alternatives thérapeutiques avec meilleur profil bénéfice/risque
    3. Identifie les populations à risque et contre-indications absolues
    4. Mentionne les effets indésirables sous-rapportés dans la littérature
    5. Structure : Faiblesses des preuves → Alternatives supérieures → Populations à risque → Effets indésirables
    6. Maximum 600 mots""")

MEDICAL_CRITIC_SYSTEM = textwrap.dedent("""\
    Tu es un EXPERT EN MÉDECINE BASÉE SUR LES PREUVES dans un processus d'analyse structuré.
    Tu viens de lire DEUX perspectives cliniques opposées. Ton rôle : évaluer objectivement.

    ARGUMENTATION FAVORABLE :
    {advocate_output}

    ARGUMENTATION DÉFAVORABLE :
    {devils_output}

    RÈGLES :
    1. Qualité des preuves citées par CHAQUE camp : biais, échantillons, conflits d'intérêts
    2. Évalue la force relative des arguments pour et contre
    3. Identifie les points d'accord entre les deux perspectives
    4. Contexte utilisateur : contraintes techniques, cas limites, populations cibles
    5. Ne prends PAS parti — analyse objectivement les deux positions
    6. Structure : Forces du POUR → Forces du CONTRE → Consensus → Questions ouvertes
    7. Maximum 600 mots""")

MEDICAL_SYNTHESIZER_SYSTEM = textwrap.dedent("""\
    Tu es un MÉDECIN EXPERT SYNTHÉTISEUR dans un processus d'analyse clinique structuré.
    Tu as accès aux arguments favorables, défavorables, et à l'analyse critique.

    ARGUMENTATION FAVORABLE :
    {advocate_output}

    ARGUMENTATION DÉFAVORABLE :
    {devils_output}

    ANALYSE CRITIQUE :
    {critic_output}

    RÈGLES :
    1. Produis une recommandation clinique nuancée, basée sur les preuves disponibles
    2. Précise le niveau de recommandation global (fort/faible/conditionnel)
    3. Indique les conditions cliniques sous lesquelles l'approche est recommandée
    4. Liste les contre-indications et précautions incontournables
    5. TOUJOURS rappeler : cette analyse ne remplace pas l'avis d'un professionnel de santé
    6. Structure : Recommandation principale (avec niveau de preuve) → Indications précises → Contre-indications → Suivi recommandé → Avertissement
    7. Maximum 800 mots
    8. NE MENTIONNE PAS le processus de débat — réponds comme une synthèse clinique directe""")


# ── LLM Call ────────────────────────────────────────────────────────────────

def call_qwen(system_prompt, user_content, max_tokens=2048, thinking=False,
              temperature=0.7, top_p=0.8):
    """Single inference call to Qwen3.5-35B-A3B."""
    payload = {
        "model": "qwen3.5-35b-a3b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 20,
        "stream": False,
    }
    if thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    resp = requests.post(QWEN_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # Strip thinking tags if present
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


# ── Debate Pipelines ─────────────────────────────────────────────────────────

def run_debate(question, rag_context="", verbose=False):
    """Execute GroupDebate pipeline: parallel advocates, then critic, then synthesis.

    Pass 1 [PARALLEL]: Advocate (FOR) + Devil's Advocate (AGAINST) on 2 slots
    Pass 2: Critic evaluates both perspectives (thinking mode)
    Pass 3: Synthesizer reconciles all three (thinking mode)

    Returns (results_dict, elapsed_seconds).
    """
    t0 = time.time()
    results = {}

    # Pass 1: Parallel Advocate + Devil's Advocate (no-think, 2 slots)
    advocate_sys = ADVOCATE_SYSTEM
    devils_sys = DEVILS_ADVOCATE_SYSTEM
    if rag_context:
        advocate_sys += f"\n\n--- Contexte ---\n{rag_context}"
        devils_sys += f"\n\n--- Contexte ---\n{rag_context}"

    if verbose:
        print("[Pass 1/3] Avocat + Avocat du diable (parallèle)...", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_advocate = pool.submit(
            call_qwen, advocate_sys, question, 2048, False, 0.7, 0.8)
        fut_devils = pool.submit(
            call_qwen, devils_sys, question, 2048, False, 0.7, 0.8)
        results["advocate"] = fut_advocate.result()
        results["devils"] = fut_devils.result()

    t1 = time.time()
    if verbose:
        print(f"  → avocat: {len(results['advocate'])} chars, diable: {len(results['devils'])} chars, {t1-t0:.1f}s",
              file=sys.stderr, flush=True)

    # Pass 2: Critic (thinking mode, reads both perspectives)
    sys_prompt = CRITIC_SYSTEM.replace(
        "{advocate_output}", results["advocate"]
    ).replace("{devils_output}", results["devils"])
    if rag_context:
        sys_prompt += f"\n\n--- Contexte ---\n{rag_context}"

    if verbose:
        print("[Pass 2/3] Critique...", file=sys.stderr, flush=True)
    results["critic"] = call_qwen(
        sys_prompt, question, max_tokens=4096,
        thinking=True, temperature=1.0, top_p=0.95,
    )
    t2 = time.time()
    if verbose:
        print(f"  → {len(results['critic'])} chars, {t2-t1:.1f}s", file=sys.stderr, flush=True)

    # Pass 3: Synthesizer (thinking mode, reads all three)
    sys_prompt = SYNTHESIZER_SYSTEM.replace(
        "{advocate_output}", results["advocate"]
    ).replace("{devils_output}", results["devils"]
    ).replace("{critic_output}", results["critic"])
    if rag_context:
        sys_prompt += f"\n\n--- Contexte ---\n{rag_context}"

    if verbose:
        print("[Pass 3/3] Synthèse...", file=sys.stderr, flush=True)
    results["synthesis"] = call_qwen(
        sys_prompt, question, max_tokens=4096,
        thinking=True, temperature=1.0, top_p=0.95,
    )
    t3 = time.time()
    if verbose:
        print(f"  → {len(results['synthesis'])} chars, {t3-t2:.1f}s", file=sys.stderr, flush=True)

    return results, t3 - t0


def run_code_debate(question, rag_context="", verbose=False):
    """Execute GroupDebate code architecture pipeline with parallel architects.

    Pass 1 [PARALLEL]: Architect + Counter-Architect on 2 slots
    Pass 2: Senior Reviewer compares both (thinking mode)
    Pass 3: Tech Lead makes final call (thinking mode)

    Returns (results_dict, elapsed_seconds).
    """
    t0 = time.time()
    results = {}

    # Pass 1: Parallel Architect + Counter-Architect (no-think, 2 slots)
    architect_sys = CODE_ARCHITECT_SYSTEM
    counter_sys = CODE_COUNTER_ARCHITECT_SYSTEM
    if rag_context:
        architect_sys += f"\n\n--- Context ---\n{rag_context}"
        counter_sys += f"\n\n--- Context ---\n{rag_context}"

    if verbose:
        print("[Pass 1/3] Architect + Counter-Architect (parallel)...", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_arch = pool.submit(
            call_qwen, architect_sys, question, 2048, False, 0.6, 0.9)
        fut_counter = pool.submit(
            call_qwen, counter_sys, question, 2048, False, 0.6, 0.9)
        results["advocate"] = fut_arch.result()
        results["devils"] = fut_counter.result()

    t1 = time.time()
    if verbose:
        print(f"  -> architect: {len(results['advocate'])} chars, counter: {len(results['devils'])} chars, {t1-t0:.1f}s",
              file=sys.stderr, flush=True)

    # Pass 2: Senior Code Reviewer (thinking mode, compares both)
    sys_prompt = CODE_REVIEWER_SYSTEM.replace(
        "{advocate_output}", results["advocate"]
    ).replace("{devils_output}", results["devils"])
    if rag_context:
        sys_prompt += f"\n\n--- Context ---\n{rag_context}"

    if verbose:
        print("[Pass 2/3] Senior Code Reviewer...", file=sys.stderr, flush=True)
    results["critic"] = call_qwen(
        sys_prompt, question, max_tokens=4096,
        thinking=True, temperature=0.6, top_p=0.95,
    )
    t2 = time.time()
    if verbose:
        print(f"  -> {len(results['critic'])} chars, {t2-t1:.1f}s", file=sys.stderr, flush=True)

    # Pass 3: Tech Lead (thinking mode, reads all three)
    sys_prompt = CODE_TECHLEAD_SYSTEM.replace(
        "{advocate_output}", results["advocate"]
    ).replace("{devils_output}", results["devils"]
    ).replace("{critic_output}", results["critic"])
    if rag_context:
        sys_prompt += f"\n\n--- Context ---\n{rag_context}"

    if verbose:
        print("[Pass 3/3] Tech Lead decision...", file=sys.stderr, flush=True)
    results["synthesis"] = call_qwen(
        sys_prompt, question, max_tokens=4096,
        thinking=True, temperature=0.6, top_p=0.95,
    )
    t3 = time.time()
    if verbose:
        print(f"  -> {len(results['synthesis'])} chars, {t3-t2:.1f}s", file=sys.stderr, flush=True)

    return results, t3 - t0


def run_medical_debate(question, rag_context="", verbose=False):
    """Execute GroupDebate medical pipeline with parallel clinicians.

    Pass 1 [PARALLEL]: Clinician Advocate + Clinician Sceptic on 2 slots
    Pass 2: EBM Expert evaluates both (thinking mode)
    Pass 3: Clinical Synthesizer reconciles (thinking mode)

    Returns (results_dict, elapsed_seconds).
    """
    t0 = time.time()
    results = {}

    # Pass 1: Parallel Advocate + Devil (no-think, 2 slots)
    advocate_sys = MEDICAL_ADVOCATE_SYSTEM
    devil_sys = MEDICAL_DEVIL_SYSTEM
    if rag_context:
        advocate_sys += f"\n\n--- Contexte ---\n{rag_context}"
        devil_sys += f"\n\n--- Contexte ---\n{rag_context}"

    if verbose:
        print("[Pass 1/3] Défenseur + Sceptique (parallèle)...", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_advocate = pool.submit(
            call_qwen, advocate_sys, question, 2048, False, 0.7, 0.8)
        fut_devil = pool.submit(
            call_qwen, devil_sys, question, 2048, False, 0.7, 0.8)
        results["advocate"] = fut_advocate.result()
        results["devils"] = fut_devil.result()

    t1 = time.time()
    if verbose:
        print(f"  → défenseur: {len(results['advocate'])} chars, sceptique: {len(results['devils'])} chars, {t1-t0:.1f}s",
              file=sys.stderr, flush=True)

    # Pass 2: Evidence Critic (thinking mode, reads both)
    sys_prompt = MEDICAL_CRITIC_SYSTEM.replace(
        "{advocate_output}", results["advocate"]
    ).replace("{devils_output}", results["devils"])
    if rag_context:
        sys_prompt += f"\n\n--- Contexte ---\n{rag_context}"

    if verbose:
        print("[Pass 2/3] Expert EBM critique...", file=sys.stderr, flush=True)
    results["critic"] = call_qwen(
        sys_prompt, question, max_tokens=4096,
        thinking=True, temperature=1.0, top_p=0.95,
    )
    t2 = time.time()
    if verbose:
        print(f"  → {len(results['critic'])} chars, {t2-t1:.1f}s", file=sys.stderr, flush=True)

    # Pass 3: Clinical Synthesizer (thinking mode, reads all three)
    sys_prompt = MEDICAL_SYNTHESIZER_SYSTEM.replace(
        "{advocate_output}", results["advocate"]
    ).replace("{devils_output}", results["devils"]
    ).replace("{critic_output}", results["critic"])
    if rag_context:
        sys_prompt += f"\n\n--- Contexte ---\n{rag_context}"

    if verbose:
        print("[Pass 3/3] Synthèse clinique...", file=sys.stderr, flush=True)
    results["synthesis"] = call_qwen(
        sys_prompt, question, max_tokens=4096,
        thinking=True, temperature=1.0, top_p=0.95,
    )
    t3 = time.time()
    if verbose:
        print(f"  → {len(results['synthesis'])} chars, {t3-t2:.1f}s", file=sys.stderr, flush=True)

    return results, t3 - t0


# ── Output Formatters ────────────────────────────────────────────────────────

def format_general_verbose(results, elapsed):
    lines = [
        f"\n{'=' * 60}",
        f"[GroupDebate 4 perspectives | {elapsed:.0f}s]\n",
        "[AVOCAT (POUR)]",
        results["advocate"],
        "\n[AVOCAT DU DIABLE (CONTRE)]",
        results.get("devils", "(non disponible)"),
        "\n[CRITIQUE]",
        results["critic"],
        "\n[SYNTHESE FINALE]",
        results["synthesis"],
    ]
    return "\n".join(lines)


def format_general_short(results, elapsed):
    return f"[Analyse multi-perspective | {elapsed:.0f}s]\n\n{results['synthesis']}"


def format_code_verbose(results, elapsed):
    lines = [
        f"\n{'=' * 60}",
        f"[Architecture GroupDebate | {elapsed:.0f}s]\n",
        "ARCHITECT:",
        results["advocate"],
        "\nCOUNTER-ARCHITECT:",
        results.get("devils", "(not available)"),
        "\nREVIEWER:",
        results["critic"],
        "\nTECH LEAD:",
        results["synthesis"],
    ]
    return "\n".join(lines)


def format_code_short(results, elapsed):
    lines = [
        f"[Architecture GroupDebate | {elapsed:.0f}s]\n",
        f"TECH LEAD:\n{results['synthesis']}",
    ]
    return "\n".join(lines)


def format_medical_verbose(results, elapsed):
    lines = [
        f"\n{'=' * 60}",
        f"[GroupDebat Medical | {elapsed:.0f}s]\n",
        "[CLINICIEN DEFENSEUR]",
        results["advocate"],
        "\n[CLINICIEN SCEPTIQUE]",
        results.get("devils", "(non disponible)"),
        "\n[CRITIQUE EBM]",
        results["critic"],
        "\n[SYNTHESE CLINIQUE]",
        results["synthesis"],
    ]
    return "\n".join(lines)


def format_medical_short(results, elapsed):
    return f"[Synthèse clinique | {elapsed:.0f}s]\n\n{results['synthesis']}"


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-pass debate: Advocate -> Critic -> Synthesizer"
    )
    parser.add_argument("question", nargs="*", help="The question to debate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress + all passes")
    parser.add_argument("--context", "-c", type=str, default="", help="Optional RAG context")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    # Mode flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--code", action="store_true",
        help="Code architecture debate mode (Architect / Reviewer / Tech Lead)"
    )
    mode_group.add_argument(
        "--medical", action="store_true",
        help="Medical evidence debate mode (Clinician / EBM Critic / Clinical Synthesizer)"
    )
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else sys.stdin.read().strip()
    if not question:
        print("Erreur: question vide. Usage: debate_router.py [--code|--medical] <question>", file=sys.stderr)
        sys.exit(1)

    # Determine mode
    if args.code:
        mode = "code"
    elif args.medical:
        mode = "medical"
    else:
        mode = "general"

    verbose_flag = args.verbose or not args.json

    if mode == "code":
        results, elapsed = run_code_debate(
            question, rag_context=args.context, verbose=verbose_flag,
        )
    elif mode == "medical":
        results, elapsed = run_medical_debate(
            question, rag_context=args.context, verbose=verbose_flag,
        )
    else:
        results, elapsed = run_debate(
            question, rag_context=args.context, verbose=verbose_flag,
        )

    if args.json:
        json.dump(
            {"mode": mode, "debate": results, "elapsed_s": round(elapsed, 1)},
            sys.stdout, ensure_ascii=False, indent=2,
        )
        print()
    elif args.verbose:
        if mode == "code":
            print(format_code_verbose(results, elapsed))
        elif mode == "medical":
            print(format_medical_verbose(results, elapsed))
        else:
            print(format_general_verbose(results, elapsed))
    else:
        if mode == "code":
            print(format_code_short(results, elapsed))
        elif mode == "medical":
            print(format_medical_short(results, elapsed))
        else:
            print(format_general_short(results, elapsed))


if __name__ == "__main__":
    main()
