
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Callable, Literal
from pathlib import Path
import copy
import json
import random
import re
import uuid

WorkloadType = Literal["rag", "scientific"]
SemanticClass = Literal["meaning_preserving", "meaning_changing"]
GenerationClass = Literal["algorithmic", "llm_generated"]


@dataclass
class PromptRecord:
    prompt_id: str
    workload: WorkloadType
    semantic_class: SemanticClass
    generation_class: GenerationClass
    mutation_type: str
    mutation_severity: float
    base_prompt: str
    mutated_prompt: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGExample:
    system_instruction: str
    user_query: str
    retrieved_chunks: List[str]
    output_instruction: str
    # Ground-truth answer(s) from the source QA dataset, carried through the
    # pipeline so generated output can be scored later.
    #
    # NEVER include this in render(). The model must not see the answer; it is
    # reference-only metadata. Defaulted so JSONLs written before this field
    # existed still deserialize via RAGExample(**row).
    reference_answers: List[str] = field(default_factory=list)

    def render(self) -> str:
        chunks_text = "\n\n".join(
            [f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(self.retrieved_chunks)]
        )
        return (
            f"{self.system_instruction}\n\n"
            f"User Question:\n{self.user_query}\n\n"
            f"Retrieved Context:\n{chunks_text}\n\n"
            f"{self.output_instruction}"
        ).strip()


@dataclass
class ScientificExample:
    problem_description: str
    equation_name: str
    parameters: Dict[str, float]
    grid: Dict[str, int]
    constraints: Dict[str, str]
    output_schema: str
    render_order: Optional[List[str]] = None

    def render(self) -> str:
        sections = {
            "problem": f"Problem:\n{self.problem_description}",
            "equation": f"Equation:\n{self.equation_name}",
            "parameters": "Parameters:\n" + "\n".join([f"- {k}: {v}" for k, v in self.parameters.items()]),
            "grid": "Grid:\n" + "\n".join([f"- {k}: {v}" for k, v in self.grid.items()]),
            "constraints": "Constraints:\n" + "\n".join([f"- {k}: {v}" for k, v in self.constraints.items()]),
            "output_schema": f"Output Format:\n{self.output_schema}",
        }
        order = self.render_order or ["problem", "equation", "parameters", "grid", "constraints", "output_schema"]
        body = "\n\n".join(sections[k] for k in order)
        return f"Solve the following scientific problem.\n\n{body}".strip()


class BaseMutator:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def _shuffle_until_different(self, seq: List[Any], max_retries: int = 16) -> None:
        """Shuffle `seq` in place until its order differs from the input.

        Raises ValueError if the sequence cannot diverge (length < 2 or all
        elements identical). Falls back to reversing the sequence after
        `max_retries` failed shuffles.
        """
        if len(seq) < 2:
            raise ValueError(f"Cannot reorder a sequence of length {len(seq)}.")
        if all(x == seq[0] for x in seq):
            raise ValueError("Cannot reorder a sequence whose elements are all identical.")
        original = list(seq)
        for _ in range(max_retries):
            self.rng.shuffle(seq)
            if seq != original:
                return
        seq[:] = original[::-1]
        if seq == original:
            raise ValueError("Shuffle-until-different fallback (reverse) failed to produce a new order.")

    def mutate_rag(self, base: RAGExample, mutation_type: str, mutation_severity: float = 1.0,
                   candidate_chunks: Optional[List[str]] = None, llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        raise NotImplementedError

    def mutate_scientific(self, base: ScientificExample, mutation_type: str, mutation_severity: float = 1.0,
                          llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        raise NotImplementedError

    def _new_record(self, workload: WorkloadType, semantic_class: SemanticClass, generation_class: GenerationClass,
                    mutation_type: str, mutation_severity: float, base_prompt: str, mutated_prompt: str,
                    metadata: Dict[str, Any]) -> PromptRecord:
        return PromptRecord(
            prompt_id=str(uuid.uuid4()),
            workload=workload,
            semantic_class=semantic_class,
            generation_class=generation_class,
            mutation_type=mutation_type,
            mutation_severity=mutation_severity,
            base_prompt=base_prompt,
            mutated_prompt=mutated_prompt,
            metadata=metadata,
        )


class AlgorithmicMeaningPreservingMutator(BaseMutator):
    # Word-level synonym table used by `synonym_substitution`. Keys must be the
    # lowercase surface form and the values must be meaning-preserving in a
    # factual-QA context. The lookup is case-insensitive and the renderer
    # preserves the original word's leading capitalization.
    #
    # The original table only covered instruction-style words (`brief`,
    # `answer`, `solve`, ...) which almost never appear in natural-questions
    # user_queries, so substitution silently produced no change on 99% of
    # inputs. The expanded table below adds verbs / nouns / adjectives that
    # are common in factual question stems and includes the most-used
    # inflections of each verb. Audited on outputs/processed/rag_examples.jsonl
    # to ensure a high match rate (see prompt_mutation/test_mutations.ipynb).
    SYNONYMS = {
        # Original instruction-style pairs.
        "brief": "concise", "answer": "response", "carefully": "thoroughly",
        "main": "primary", "question": "query", "solve": "compute",
        "following": "given", "return": "produce",
        # Verbs (base / past / past-participle / 3rd-singular).
        "begin": "start", "began": "started", "begins": "starts",
        "happen": "occur", "happened": "occurred", "happens": "occurs",
        "make": "create", "made": "created", "makes": "creates",
        "find": "locate", "found": "located", "finds": "locates",
        "use": "utilize", "used": "utilized", "uses": "utilizes",
        "build": "construct", "built": "constructed", "builds": "constructs",
        "show": "display", "showed": "displayed", "shows": "displays",
        "live": "reside", "lives": "resides", "lived": "resided",
        "call": "name", "called": "named", "calls": "names",
        "get": "obtain", "got": "obtained", "gets": "obtains",
        "see": "observe", "saw": "observed", "sees": "observes",
        "die": "perish", "died": "perished", "dies": "perishes",
        "buy": "purchase", "bought": "purchased", "buys": "purchases",
        "own": "possess", "owned": "possessed", "owns": "possesses",
        "win": "triumph", "won": "triumphed", "wins": "triumphs",
        "come": "arrive", "came": "arrived", "comes": "arrives",
        "write": "author", "wrote": "authored", "writes": "authors",
        "give": "provide", "gave": "provided", "gives": "provides",
        "take": "acquire", "took": "acquired", "takes": "acquires",
        "say": "state", "said": "stated", "says": "states",
        "play": "perform", "played": "performed", "plays": "performs",
        # Common nouns.
        "country": "nation", "countries": "nations",
        "person": "individual", "people": "individuals",
        "year": "annum", "years": "annums",
        # Adjectives.
        "big": "large", "small": "tiny", "old": "aged", "new": "novel",
        "many": "numerous", "different": "distinct", "best": "finest",
        "first": "initial", "last": "final",
    }

    def mutate_rag(self, base: RAGExample, mutation_type: str, mutation_severity: float = 1.0,
                   candidate_chunks: Optional[List[str]] = None, llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        mutated = copy.deepcopy(base)
        changed_field = None

        if mutation_type == "typo":
            mutated.user_query = self._inject_typo(mutated.user_query)
            changed_field = "user_query"
        elif mutation_type == "formatting":
            mutated.output_instruction = f"### Output Requirements ###\n- {mutated.output_instruction}"
            changed_field = "output_instruction"
        elif mutation_type == "template_rewrite":
            mutated.output_instruction = self._template_rewrite(mutated.output_instruction)
            changed_field = "output_instruction"
        elif mutation_type == "synonym_substitution":
            mutated.user_query = self._synonym_substitute(mutated.user_query)
            changed_field = "user_query"
        elif mutation_type == "chunk_reorder":
            if len(mutated.retrieved_chunks) < 2:
                raise ValueError(
                    f"chunk_reorder requires >= 2 retrieved_chunks (got {len(mutated.retrieved_chunks)}). "
                    "Increase min_chunks in DataLoadConfig or use distractor padding."
                )
            self._shuffle_until_different(mutated.retrieved_chunks)
            changed_field = "retrieved_chunks"
        else:
            raise ValueError(f"Unsupported algorithmic meaning-preserving RAG mutation: {mutation_type}")

        return self._new_record("rag", "meaning_preserving", "algorithmic", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})

    def mutate_scientific(self, base: ScientificExample, mutation_type: str, mutation_severity: float = 1.0,
                          llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        mutated = copy.deepcopy(base)
        changed_field = None

        if mutation_type == "typo":
            mutated.problem_description = self._inject_typo(mutated.problem_description)
            changed_field = "problem_description"
        elif mutation_type == "formatting":
            mutated.output_schema = f"Please follow this schema exactly:\n{mutated.output_schema}"
            changed_field = "output_schema"
        elif mutation_type == "template_rewrite":
            mutated.problem_description = self._template_rewrite(mutated.problem_description)
            changed_field = "problem_description"
        elif mutation_type == "field_reorder":
            default_order = ["problem", "equation", "parameters", "grid", "constraints", "output_schema"]
            current_order = base.render_order or default_order
            new_order = list(current_order)
            self._shuffle_until_different(new_order)
            mutated.render_order = new_order
            changed_field = "render_order"
        elif mutation_type == "synonym_substitution":
            mutated.problem_description = self._synonym_substitute(mutated.problem_description)
            changed_field = "problem_description"
        else:
            raise ValueError(f"Unsupported algorithmic meaning-preserving scientific mutation: {mutation_type}")

        return self._new_record("scientific", "meaning_preserving", "algorithmic", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})

    def _inject_typo(self, text: str, max_retries: int = 16) -> str:
        """Inject a single-character edit (swap / delete / duplicate) into
        a random word. Retries with a different (word, op) pair if the edit
        produces no change (e.g. swapping two identical adjacent characters
        like 'll' in 'really'). Raises ValueError if every retry produces
        the same string, which only happens on degenerate inputs.
        """
        words = text.split()
        # Words shorter than 2 chars cannot be edited meaningfully.
        candidate_indices = [i for i, w in enumerate(words) if len(w) >= 2]
        if not candidate_indices:
            raise ValueError(
                "Cannot inject a typo: every word in the input has length < 2."
            )
        for _ in range(max_retries):
            idx = self.rng.choice(candidate_indices)
            word = words[idx]
            chars = list(word)
            op = self.rng.choice(["swap", "delete", "duplicate"])
            if op == "swap" and len(chars) >= 2:
                i = self.rng.randrange(len(chars) - 1)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "delete":
                i = self.rng.randrange(len(chars))
                del chars[i]
            else:
                i = self.rng.randrange(len(chars))
                chars.insert(i, chars[i])
            new_word = "".join(chars)
            if new_word != word:
                out = list(words)
                out[idx] = new_word
                return " ".join(out)
        raise ValueError(
            f"Failed to inject a typo after {max_retries} retries; input "
            "likely contains only words with identical adjacent characters."
        )

    def _template_rewrite(self, text: str) -> str:
        rewrites = {
            "Answer briefly": "Provide a concise answer",
            "Answer in a short paragraph": "Respond with a short paragraph",
            "Return JSON": "Produce a JSON output",
        }
        out = text
        for src, dst in rewrites.items():
            out = out.replace(src, dst)
        return out

    def _synonym_substitute(self, text: str) -> str:
        """Substitute every word that appears as a key in SYNONYMS with its
        value, preserving the original capitalization. Raises ValueError if
        no key matched any word in the input, because the caller has no way
        to recover from a silent no-op (the downstream save would reject
        base == mutated). The build script catches this and skips the
        example so the rest of the dataset survives.
        """
        def repl(match):
            word = match.group(0)
            lower = word.lower()
            if lower in self.SYNONYMS:
                new_word = self.SYNONYMS[lower]
                return new_word.capitalize() if word[0].isupper() else new_word
            return word
        new_text = re.sub(r"\b[A-Za-z]+\b", repl, text)
        if new_text == text:
            raise ValueError(
                "synonym_substitution: no SYNONYMS key matched any word in "
                f"the input ({len(self.SYNONYMS)} keys tried). Consider "
                "adding more pairs to AlgorithmicMeaningPreservingMutator.SYNONYMS."
            )
        return new_text


class AlgorithmicMeaningChangingMutator(BaseMutator):
    def mutate_rag(self, base: RAGExample, mutation_type: str, mutation_severity: float = 1.0,
                   candidate_chunks: Optional[List[str]] = None, llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        mutated = copy.deepcopy(base)
        changed_field = None

        if mutation_type == "document_replacement":
            if not mutated.retrieved_chunks:
                raise ValueError("No retrieved chunks available.")
            idx = self.rng.randrange(len(mutated.retrieved_chunks))
            mutated.retrieved_chunks[idx] = self._sample_replacement_chunk(candidate_chunks, mutated.retrieved_chunks[idx] + " [REPLACED]")
            changed_field = "retrieved_chunks"
        elif mutation_type == "document_insertion":
            new_chunk = self._sample_replacement_chunk(candidate_chunks, "Additional conflicting evidence.")
            pos = self.rng.randrange(len(mutated.retrieved_chunks) + 1)
            mutated.retrieved_chunks.insert(pos, new_chunk)
            changed_field = "retrieved_chunks"
        elif mutation_type == "query_target_change":
            mutated.user_query = self._query_target_change(mutated.user_query)
            changed_field = "user_query"
        elif mutation_type == "time_reference_change":
            mutated.user_query = self._time_reference_change(mutated.user_query)
            changed_field = "user_query"
        elif mutation_type in {"negation_flip", "agreement_flip", "polarity_flip"}:
            mutated.user_query = self._polarity_change(mutated.user_query, mutation_type)
            changed_field = "user_query"
        else:
            raise ValueError(f"Unsupported algorithmic meaning-changing RAG mutation: {mutation_type}")

        return self._new_record("rag", "meaning_changing", "algorithmic", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})

    def mutate_scientific(self, base: ScientificExample, mutation_type: str, mutation_severity: float = 1.0,
                          llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        mutated = copy.deepcopy(base)
        changed_field = None

        if mutation_type == "parameter_change":
            key = self.rng.choice(list(mutated.parameters.keys()))
            old = mutated.parameters[key]
            mutated.parameters[key] = round(old * (1.0 + 0.2 * mutation_severity), 6)
            changed_field = f"parameters.{key}"
        elif mutation_type == "unit_change":
            mutated.problem_description = self._unit_change(mutated.problem_description)
            changed_field = "problem_description"
        elif mutation_type == "constraint_change":
            key = self.rng.choice(list(mutated.constraints.keys()))
            mutated.constraints[key] = f"{mutated.constraints[key]} (modified)"
            changed_field = f"constraints.{key}"
        elif mutation_type == "grid_change":
            key = self.rng.choice(list(mutated.grid.keys()))
            mutated.grid[key] = max(1, int(round(mutated.grid[key] * (1.0 + 0.25 * mutation_severity))))
            changed_field = f"grid.{key}"
        elif mutation_type in {"negation_flip", "agreement_flip", "polarity_flip"}:
            mutated.problem_description = self._polarity_change(mutated.problem_description, mutation_type)
            changed_field = "problem_description"
        else:
            raise ValueError(f"Unsupported algorithmic meaning-changing scientific mutation: {mutation_type}")

        return self._new_record("scientific", "meaning_changing", "algorithmic", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})

    def _sample_replacement_chunk(self, candidate_chunks: Optional[List[str]], fallback: str) -> str:
        if candidate_chunks:
            return self.rng.choice(candidate_chunks)
        return fallback

    def _query_target_change(self, query: str) -> str:
        replacements = [("today", "tomorrow"), ("largest", "smallest"), ("increase", "decrease")]
        out = query
        for a, b in replacements:
            if a in out.lower():
                return re.sub(a, b, out, flags=re.IGNORECASE)
        return query + " in Europe"

    def _time_reference_change(self, query: str) -> str:
        if "today" in query.lower():
            return re.sub("today", "tomorrow", query, flags=re.IGNORECASE)
        return query + " tomorrow"

    def _unit_change(self, text: str) -> str:
        replacements = [("meters", "centimeters"), ("meter", "centimeter"), ("Celsius", "Fahrenheit"), ("seconds", "milliseconds")]
        out = text
        for a, b in replacements:
            if a in out:
                return out.replace(a, b)
        return out + " Use centimeters instead of meters."

    def _polarity_change(self, text: str, mutation_type: str) -> str:
        out = text
        if mutation_type == "negation_flip":
            pairs = [("must ", "must not "), ("should ", "should not "), ("can ", "cannot "), ("is ", "is not ")]
            for a, b in pairs:
                if a in out.lower():
                    return re.sub(a, b, out, flags=re.IGNORECASE, count=1)
            return "Do not " + out
        if mutation_type == "agreement_flip":
            pairs = [("agree", "disagree"), ("support", "oppose"), ("accept", "reject"), ("include", "exclude")]
            for a, b in pairs:
                if a in out.lower():
                    return re.sub(a, b, out, flags=re.IGNORECASE, count=1)
            return out + " Disagree with the previous assumption."
        if mutation_type == "polarity_flip":
            pairs = [("positive", "negative"), ("increase", "decrease"), ("maximize", "minimize"), ("higher", "lower")]
            for a, b in pairs:
                if a in out.lower():
                    return re.sub(a, b, out, flags=re.IGNORECASE, count=1)
            return out + " Use the opposite polarity."
        return out


class LLMMeaningPreservingMutator(BaseMutator):
    def mutate_rag(self, base: RAGExample, mutation_type: str, mutation_severity: float = 1.0,
                   candidate_chunks: Optional[List[str]] = None, llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        if llm_fn is None:
            raise ValueError("llm_fn is required for LLM-generated mutations.")
        mutated = copy.deepcopy(base)
        if mutation_type == "query_rephrase":
            prompt = (
                "Rewrite the following user query so that it preserves the exact meaning "
                "and expected answer, but changes the wording.\n\n"
                f"Query: {base.user_query}"
            )
            mutated.user_query = llm_fn(prompt)
            changed_field = "user_query"
        elif mutation_type == "chunk_paraphrase":
            if not mutated.retrieved_chunks:
                raise ValueError("No retrieved chunks available.")
            idx = self.rng.randrange(len(mutated.retrieved_chunks))
            prompt = (
                "Paraphrase the following passage while preserving all factual content.\n\n"
                f"Passage: {mutated.retrieved_chunks[idx]}"
            )
            mutated.retrieved_chunks[idx] = llm_fn(prompt)
            changed_field = f"retrieved_chunks[{idx}]"
        else:
            raise ValueError(f"Unsupported LLM meaning-preserving RAG mutation: {mutation_type}")
        return self._new_record("rag", "meaning_preserving", "llm_generated", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})

    def mutate_scientific(self, base: ScientificExample, mutation_type: str, mutation_severity: float = 1.0,
                          llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        if llm_fn is None:
            raise ValueError("llm_fn is required for LLM-generated mutations.")
        mutated = copy.deepcopy(base)
        if mutation_type == "problem_rephrase":
            prompt = (
                "Rewrite the following scientific problem description so that the task, "
                "parameters, and intended output remain the same.\n\n"
                f"Problem: {base.problem_description}"
            )
            mutated.problem_description = llm_fn(prompt)
            changed_field = "problem_description"
        else:
            raise ValueError(f"Unsupported LLM meaning-preserving scientific mutation: {mutation_type}")
        return self._new_record("scientific", "meaning_preserving", "llm_generated", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})


class LLMMeaningChangingMutator(BaseMutator):
    def mutate_rag(self, base: RAGExample, mutation_type: str, mutation_severity: float = 1.0,
                   candidate_chunks: Optional[List[str]] = None, llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        if llm_fn is None:
            raise ValueError("llm_fn is required for LLM-generated mutations.")
        mutated = copy.deepcopy(base)
        if mutation_type == "query_change":
            prompt = (
                "Rewrite the following user query so that it stays close in style and structure, "
                "but asks for a meaningfully different answer.\n\n"
                f"Query: {base.user_query}"
            )
            mutated.user_query = llm_fn(prompt)
            changed_field = "user_query"
        else:
            raise ValueError(f"Unsupported LLM meaning-changing RAG mutation: {mutation_type}")
        return self._new_record("rag", "meaning_changing", "llm_generated", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})

    def mutate_scientific(self, base: ScientificExample, mutation_type: str, mutation_severity: float = 1.0,
                          llm_fn: Optional[Callable[[str], str]] = None) -> PromptRecord:
        if llm_fn is None:
            raise ValueError("llm_fn is required for LLM-generated mutations.")
        mutated = copy.deepcopy(base)
        if mutation_type == "scenario_change":
            prompt = (
                "Rewrite the following scientific problem so that it is structurally similar, "
                "but changes a meaningful scientific condition, parameter regime, or requested outcome.\n\n"
                f"Problem: {base.problem_description}"
            )
            mutated.problem_description = llm_fn(prompt)
            changed_field = "problem_description"
        else:
            raise ValueError(f"Unsupported LLM meaning-changing scientific mutation: {mutation_type}")
        return self._new_record("scientific", "meaning_changing", "llm_generated", mutation_type, mutation_severity,
                                base.render(), mutated.render(), {"changed_field": changed_field})


def save_prompt_records(
    records: List[PromptRecord],
    root_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    if root_dir is None:
        root_dir = str(Path(__file__).resolve().parents[1] / "outputs" / "mutation")
    if not records:
        raise ValueError("No records to save.")
    degenerate = [r.prompt_id for r in records if r.base_prompt == r.mutated_prompt]
    if degenerate:
        raise ValueError(
            f"{len(degenerate)} record(s) have base_prompt == mutated_prompt "
            f"(first: {degenerate[0]}). The mutation failed to produce a different prompt; "
            "fix the mutator or the upstream data instead of saving."
        )
    first = records[0]
    bucket_dir = Path(root_dir) / first.workload / first.semantic_class / first.generation_class / first.mutation_type
    bucket_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{uuid.uuid4().hex}.jsonl"
    out_path = bucket_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return out_path
