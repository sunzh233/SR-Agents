"""ReAct engine for ToolQA.

Implements a one-LLM-call-per-step ReAct loop. The model emits
``Thought N: …\\nAction N: <action>`` lines and the environment produces
``Observation N: <obs>`` lines the model sees on the next step.

Two modes are exposed as separate engines:

* ``react`` — skills (if any) are injected into the system prompt; the
  action vocabulary does not include skill loading.
* ``react_progressive_disclosure`` — skills are exposed as candidates via
  a ``LoadSkill[index]`` action; the model may load zero or more during
  a trajectory.

Ported from https://github.com/night-chen/ToolQA
(``benchmark/ReAct/code/agents_chatgpt.py``).

``raw_output`` (on the returned :class:`~sragents.infer.base.InferenceResult`)
contains only model-generated tokens — Thought/Action lines and the final
Answer on Finish. The full scratchpad with injected Observation lines is
returned separately as ``transcript``.
"""

import re
import threading

from sragents.config import EXTERNAL_DIR
from sragents.corpus import display_name, load_corpus_dict
from sragents.infer.base import InferenceResult, register_engine
from sragents.llm import chat_with_metadata, get_extra_body, strip_think_tags
from sragents.prompts import build_prompt
from sragents.toolqa import ToolEnvironment, parse_action

_MAX_OBS_CHARS = 3000
_MAX_THOUGHT_CHARS = 4000


def _is_context_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "maximum context length" in message
        or "context length exceeded" in message
        or ("input tokens" in message and "max_tokens" in message)
    )


class ReActAgent:
    """Inner loop — shared between ``react`` and ``react_progressive_disclosure``."""

    def __init__(
        self,
        question: str,
        tools: ToolEnvironment,
        client,
        model: str,
        examples: str,
        max_steps: int = 20,
        max_tokens: int = 512,
        total_max_tokens: int | None = None,
        skills: list[str] | None = None,
        thinking: bool = False,
        candidate_skills: list[dict] | None = None,
        corpus: dict | None = None,
        temperature: float = 0.7,
        base_model: str | None = None,
    ):
        self.question = question
        self.temperature = temperature
        self.tools = tools
        self.client = client
        self.model = model
        self.base_model = base_model or model
        self.examples = examples
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.total_max_tokens = total_max_tokens
        self.skills = skills
        self.thinking = thinking

        self.candidate_skills = candidate_skills
        self.corpus = corpus or {}
        self.loaded_skill_ids: list[str] = []
        self._idx_map = (
            {str(i): s["skill_id"] for i, s in enumerate(candidate_skills)}
            if candidate_skills else {}
        )

        self.scratchpad = ""           # full (prompt re-feed + transcript)
        self.model_scratchpad = ""     # model tokens only (evaluator input)
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.completion_tokens = 0
        self.truncated = False
        self.limit_exceeded = False
        self.stop_reason: str | None = None

    def run(self) -> None:
        while (
            not self.finished
            and not self.truncated
            and self.step_n <= self.max_steps
        ):
            self._step()

    def _step(self) -> None:
        system, user = self._build_prompt()
        extra = get_extra_body(self.base_model, thinking=self.thinking)

        basename = self.base_model.lower().rsplit("/", 1)[-1]
        stop = None if "gpt-5" in basename else [f"\nObservation {self.step_n}:"]

        request_tokens = self.max_tokens
        if self.total_max_tokens is not None:
            remaining = self.total_max_tokens - self.completion_tokens
            if remaining <= 0:
                self.truncated = True
                self.limit_exceeded = True
                self.stop_reason = "total_output_limit"
                return
            request_tokens = min(request_tokens, remaining)

        try:
            result = chat_with_metadata(
                self.client, self.model, user, system=system,
                temperature=self.temperature, max_tokens=request_tokens,
                stop=stop, extra_body=extra,
            )
        except Exception as error:
            if not _is_context_limit_error(error):
                raise
            self.truncated = True
            self.limit_exceeded = True
            self.stop_reason = "context_limit"
            return

        if result.completion_tokens is None:
            raise RuntimeError(
                "ReAct token budgeting requires completion_tokens in the "
                "OpenAI-compatible response usage"
            )
        self.completion_tokens += result.completion_tokens
        response = result.content
        thought, action = self._parse_response(response)

        step_text = (
            f"\nThought {self.step_n}: {thought}"
            f"\nAction {self.step_n}: {action}"
        )
        self.model_scratchpad += step_text
        self.scratchpad += step_text

        if result.finish_reason == "length":
            self.truncated = True
            self.limit_exceeded = True
            self.stop_reason = "step_output_limit"
            self.step_n += 1
            return

        self.scratchpad += f"\nObservation {self.step_n}: "

        if not action.strip():
            if not thought.strip():
                self.scratchpad += (
                    "Your response was empty. This can happen when "
                    "reasoning consumed the full output budget. "
                    "Please provide a concise Thought and a specific Action."
                )
            elif len(thought) >= _MAX_THOUGHT_CHARS:
                self.scratchpad += (
                    "Your thought was truncated and the action was missing. "
                    "Please keep thoughts concise and provide a specific Action."
                )
            else:
                self.scratchpad += (
                    "Your action could not be parsed. "
                    "Please provide a valid Action on the next line."
                )
        else:
            action_type, argument = parse_action(action)

            if action_type == "Finish":
                self.answer = argument or ""
                self.scratchpad += f"Answer: {self.answer}"
                self.model_scratchpad += f"\nAnswer: {self.answer}"
                self.finished = True
            elif action_type == "LoadSkill":
                obs = self._handle_load_skill(argument)
                self.scratchpad += self._truncate_obs(obs)
            else:
                obs = self.tools.execute(action)
                self.scratchpad += self._truncate_obs(obs)

        self.step_n += 1
        if (
            not self.finished
            and self.total_max_tokens is not None
            and self.completion_tokens >= self.total_max_tokens
        ):
            self.truncated = True
            self.limit_exceeded = True
            self.stop_reason = "total_output_limit"

    def _build_prompt(self) -> tuple[str, str]:
        inst = {"dataset": "toolqa", "question": self.question}

        if self.candidate_skills is not None:
            system, base_user = build_prompt(inst)
            skill_lines = [
                f"{i} — {display_name(s, i)} — {s.get('description', '')}"
                for i, s in enumerate(self.candidate_skills)
            ]
            system += (
                "\n(14) LoadSkill[index], which loads a skill document "
                "that provides precise methodology and step-by-step "
                "procedures for a specific problem type — these often "
                "contain critical details that general knowledge may "
                "miss. For example: LoadSkill[0]"
                "\n\nAvailable skills:\n" + "\n".join(skill_lines)
            )
        else:
            system, base_user = build_prompt(inst, skills=self.skills)

        user = (
            f"Here are some examples:\n{self.examples}\n"
            f"(END OF EXAMPLES)\n"
            f"{base_user}"
            f"{self.scratchpad}\n"
            f"Thought {self.step_n}:"
        )
        return system, user

    def _handle_load_skill(self, token: str) -> str:
        if not token:
            return "LoadSkill requires an index argument."

        # Scoped to the candidate set shown to the model.
        candidates = {
            sid: self.corpus[sid]
            for sid in self._idx_map.values()
            if sid in self.corpus
        }
        skill: dict | None = None
        real_id = self._idx_map.get(token)
        if real_id:
            skill = self.corpus.get(real_id)
        if skill is None and token in candidates:
            skill = candidates[token]
        if skill is None:
            token_lower = token.lower()
            for s in candidates.values():
                if s.get("name", "").lower() == token_lower:
                    skill = s
                    break
        if skill is None:
            return f"Skill '{token}' not found. Continue solving the problem."

        self.loaded_skill_ids.append(skill["skill_id"])
        return (
            f"Skill loaded: {display_name(skill)}\n"
            f"---\n{skill.get('content', '')}\n---\n"
            f"Continue solving the problem."
        )

    @staticmethod
    def _truncate_action(action_text: str) -> str:
        """Truncate action text at the next ReAct structural marker.

        Unlike a simple split("\\n")[0], this preserves multi-line content
        inside tool calls (e.g. PythonInterpreter[code with newlines])
        while still stopping at genuine Thought/Action/Observation markers.
        """
        action_text = action_text.strip()
        # Match a line that is purely a structural marker (not inside code)
        cont = re.search(
            r"\n\s*(?:Observation|Thought|Action)\s*\d*\s*:", action_text
        )
        if cont:
            return action_text[: cont.start()].rstrip()
        return action_text

    def _parse_response(self, response: str) -> tuple[str, str]:
        response = strip_think_tags(response).strip()

        # Thinking mode can consume all tokens in <think>, leaving an empty
        # response after stripping. Surface this explicitly so the recovery
        # observation in _step can give a targeted hint.
        if not response:
            return "", ""

        # If the model self-generates an "Observation N:" line (e.g. because
        # the server ignored the stop token), truncate there so the
        # fabricated observation never contaminates raw_output.
        obs_split = re.search(r"\n\s*Observation\s*\d*\s*:", response)
        if obs_split:
            response = response[: obs_split.start()].rstrip()

        # Prefer the expected step number, but accept a reset or omitted number.
        # Step numbering is presentation generated by the model; the agent owns
        # the actual step state.
        marker = re.search(
            rf"(?m)^[ \t]*Action[ \t]*{self.step_n}[ \t]*:[ \t]*",
            response,
        )
        if marker is None:
            marker = re.search(
                r"(?m)^[ \t]*Action[ \t]*\d*[ \t]*:[ \t]*",
                response,
            )

        thought_text = response[: marker.start()] if marker else response
        thought = re.sub(
            r"^Thought[ \t]*\d*[ \t]*:[ \t]*",
            "",
            thought_text.strip(),
        ).strip().replace("\n", " ")
        action = (
            self._truncate_action(response[marker.end():])
            if marker else ""
        )

        # Truncate overly long thoughts to prevent one bad debugging spiral
        # from bloating the scratchpad for every subsequent step.
        if len(thought) > _MAX_THOUGHT_CHARS:
            thought = thought[:_MAX_THOUGHT_CHARS] + "..."

        return thought, action

    @staticmethod
    def _truncate_obs(obs: str) -> str:
        if len(obs) <= _MAX_OBS_CHARS:
            return obs
        return obs[:_MAX_OBS_CHARS] + f"... (truncated, {len(obs)} chars total)"

    def is_halted(self) -> bool:
        return self.step_n > self.max_steps and not self.finished


# A ReAct engine's configured max_tokens is the whole-trajectory output budget.
# Each individual call stays bounded so a growing scratchpad retains context
# headroom. Thinking mode needs more room for its hidden block.
_STEP_TOKENS = 6144
_STEP_TOKENS_THINKING = 16384
_MAX_STEPS = 20


class _BaseReActEngine:
    """Base class sharing the ToolEnvironment + agent construction."""

    _USE_PROGRESSIVE_DISCLOSURE: bool = False

    def __init__(
        self,
        max_steps: int = _MAX_STEPS,
        max_tokens: int | None = None,
        thinking: bool = False,
        toolqa_data_dir: str | None = None,
        temperature: float = 0.7,
    ):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self._toolqa_data_dir = toolqa_data_dir or str(EXTERNAL_DIR / "toolqa")
        self._local = threading.local()

    def _get_tools(self) -> ToolEnvironment:
        if not hasattr(self._local, "tools"):
            self._local.tools = ToolEnvironment(self._toolqa_data_dir)
        else:
            self._local.tools.reset()
        return self._local.tools

    def run(
        self,
        instance: dict,
        skills: list[dict],
        client,
        model: str,
        **kwargs,
    ) -> InferenceResult:
        from sragents.toolqa.fewshots import TOOLQA_EXAMPLES

        step_tokens = _STEP_TOKENS_THINKING if self.thinking else _STEP_TOKENS
        corpus = kwargs.get("corpus") or (
            load_corpus_dict() if self._USE_PROGRESSIVE_DISCLOSURE else {}
        )

        if self._USE_PROGRESSIVE_DISCLOSURE:
            agent = ReActAgent(
                question=instance["question"],
                tools=self._get_tools(),
                client=client, model=model,
                examples=TOOLQA_EXAMPLES,
                base_model=kwargs.get("base_model", model),
                max_steps=self.max_steps, max_tokens=step_tokens,
                total_max_tokens=self.max_tokens,
                thinking=self.thinking,
                candidate_skills=skills, corpus=corpus,
                temperature=self.temperature,
            )
        else:
            skill_texts = [s["content"] for s in skills if s.get("content")]
            agent = ReActAgent(
                question=instance["question"],
                tools=self._get_tools(),
                client=client, model=model,
                examples=TOOLQA_EXAMPLES,
                base_model=kwargs.get("base_model", model),
                max_steps=self.max_steps, max_tokens=step_tokens,
                total_max_tokens=self.max_tokens,
                skills=skill_texts or None,
                thinking=self.thinking,
                temperature=self.temperature,
            )

        agent.run()

        return InferenceResult(
            raw_output=agent.model_scratchpad,
            transcript=agent.scratchpad,
            skill_ids_used=(
                agent.loaded_skill_ids if self._USE_PROGRESSIVE_DISCLOSURE
                else [s["skill_id"] for s in skills]
            ),
            meta={
                "n_steps": agent.step_n - 1,
                "finished": agent.finished,
                "halted": agent.is_halted(),
                "failed": agent.truncated or agent.is_halted(),
                "truncated": agent.truncated,
                "limit_exceeded": agent.limit_exceeded,
                "stop_reason": (
                    agent.stop_reason
                    or ("max_steps" if agent.is_halted() else None)
                ),
                "completion_tokens": agent.completion_tokens,
                "max_tokens": agent.total_max_tokens,
            },
        )


@register_engine("react")
class ReActEngine(_BaseReActEngine):
    """ReAct with skills injected into the system prompt (ToolQA default)."""
    _USE_PROGRESSIVE_DISCLOSURE = False


@register_engine("react_progressive_disclosure")
class ReActProgressiveDisclosureEngine(_BaseReActEngine):
    """ReAct + LoadSkill action: model loads skills mid-trajectory."""
    _USE_PROGRESSIVE_DISCLOSURE = True
