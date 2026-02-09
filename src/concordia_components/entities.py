import functools

from typing import Any
from collections.abc import Sequence
from concordia.typing import entity
from typing_extensions import override
from concordia.typing.entity import ActionSpec, DEFAULT_ACTION_SPEC
from concordia.language_model.language_model import LanguageModel
from concordia_components.type_aliases import Thread

class NewsSource(entity.EntityWithLogging):
    """A news source that posts messages from a feed."""

    def __init__(self,
                 name: str,
                 news_feed: list[dict[str, str]],
                 logs: dict[str, Any] = None
                 ):
        self._name = name
        self._news_feed = news_feed
        self._current_index = 0
        self._logs = logs if logs is not None else {}

    @override
    @functools.cached_property
    def name(self) -> str:
        """The name of the entity."""
        return self._name

    @override
    def act(self, action_spec: ActionSpec = DEFAULT_ACTION_SPEC) -> str:
        """Returns the next news item."""
        if self._current_index >= len(self._news_feed):
            return ""
        
        item = self._news_feed[self._current_index]
        self._current_index += 1
        
        news = item.get('message', '')
        return news

    @override
    def observe(self, thread: Thread) -> None:
        """News source ignores observations."""
        pass

    @override
    def get_last_log(self) -> dict[str, Any]:
        """Returns debugging information."""
        return self._logs
    
    def has_news(self) -> bool:
        """Check if there are more news items."""
        return self._current_index < len(self._news_feed)

class User(entity.EntityWithLogging):
    """Base class for users.

    Entities are the basic building blocks of a game. They are the entities
    that the game master explicitly keeps track of. Entities can be anything,
    from the player's character to an inanimate object. At its core, an entity
    is an entity that has a name, can act, and can observe.

    Entities are sent observations by the game master, and they can be asked to
    act by the game master. Multiple observations can be sent to an entity before
    a request for an action attempt is made. The entities are responsible for
    keeping track of their own state, which might change upon receiving
    observations or acting.
    """

    def __init__(self,
                 model: LanguageModel,
                 name: str,
                 model_id: int = 0,
                 context: list[dict[str, str]] = None,
                 action_classifier = None,
                 logs: dict[str, Any] = None
                 ):
        """Initialize the user."""
        self._model = model
        self._name = name
        self._model_id = model_id
        self._context = context if context is not None else []
        self._action_classifier = action_classifier
        self._logs = logs if logs is not None else {}
        self._max_messages = 50
        self._max_length = 4096
        self._pending_prompt = None

    @override
    @functools.cached_property
    def name(self) -> str:
        """The name of the entity."""
        return self._name

    @property
    def model_id(self) -> int:
        return self._model_id

    def get_prompt(self, action_spec: ActionSpec = DEFAULT_ACTION_SPEC) -> str:
        """Constructs the prompt but does not sample."""
        if len(self._context) > self._max_messages:
            self._context = self._context[-self._max_messages:]
        
        while True:
            prompt = self._model.apply_chat_template(self._context, add_generation_prompt=True)
            if len(prompt) < self._max_length:
                break
            self._context = self._context[5:]

        return prompt

    def complete_action(self, response: str) -> str:
        """Completes the action with the generated response."""
        self._context.append({"role": "assistant", "content": response})
        return response

    @override
    def act(self, action_spec: ActionSpec = DEFAULT_ACTION_SPEC) -> str:
        """Returns the entity's intended action given the action spec."""
        prompt = self.get_prompt(action_spec)
        response = self._model.sample_text(prompt=prompt, max_tokens=200)
        print(f"Entity {self._name} generated response: {response}")
        exit(0)
        return self.complete_action(response)

    @override
    def observe(self, thread: Thread) -> None:
        """Informs the Entity of an observation."""
        self._context.append({"role": "system", "content": "### New Thread ###\nWrite a post for a new conversation thread."})
        self._context.extend(thread.content)

    @override
    def get_last_log(self) -> dict[str, Any]:
        """Returns debugging information in the form of a dictionary."""
        return self._logs

    def survey_response(self, question: str, options: Sequence[str]) -> str:
        """Returns the entity's response to a survey question."""
        temp_context = list(self._context)
        if len(temp_context) > self._max_messages:
            temp_context = temp_context[-self._max_messages:]
        
        temp_context.append({"role": "user", "content": question})
        while True:
            prompt = self._model.apply_chat_template(temp_context, add_generation_prompt=True)
            if len(prompt) < self._max_length:
                break

            temp_context = temp_context[5:]

        idx, choice, _ = self._model.sample_choice(prompt=prompt, responses=options)
        return choice
