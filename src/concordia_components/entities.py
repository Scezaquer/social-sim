import functools

from typing import Any
from concordia.typing import entity
from typing_extensions import override
from concordia.typing.entity import ActionSpec, DEFAULT_ACTION_SPEC
from concordia.language_model.language_model import LanguageModel
from concordia_components.type_aliases import Thread

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
                 context: str = "",
                 logs: dict[str, Any] = None
                 ):
        """Initialize the user."""
        self._model = model
        self._name = name
        self._context = context
        self._logs = logs if logs is not None else {}

    @override
    @functools.cached_property
    def name(self) -> str:
        """The name of the entity."""
        return self._name

    def _format_message(self, role: str, message: str) -> str:
        """Format a message for the user."""
        return f"<|im_start|>{role}\n{message}<|im_end|>\n"

    def _format_thread(self, thread: Thread) -> str:
        """Format a thread for the user."""
        formatted_thread = ""
        for message in thread.content:
            formatted_thread += self._format_message(message['role'], message['content'])
        return formatted_thread

    @override
    def act(self, action_spec: ActionSpec = DEFAULT_ACTION_SPEC) -> str:
        """Returns the entity's intended action given the action spec.

        Args:
        action_spec: The specification of the action that the entity is queried
            for. This might be a free-form action, a multiple choice action, or
            a float action. The action will always be a string, but it should be
            compliant with the specification.

        Returns:
        The entity's intended action.
        """
        self._context += f"<|im_start|>assistant\n"
        response = self._model.sample_text(prompt=self._context)
        self._context += f"{response}<|im_end|>\n"
        return response

    @override
    def observe(self, observation: list[Thread]) -> None:
        """Informs the Entity of an observation.

        Args:
        observation: The observation for the entity to process. Always a string.
        """
        for thread in observation:
            self._context += "### New Thread ###\n"
            self._context += self._format_thread(thread)

    @override
    def get_last_log(self) -> dict[str, Any]:
        """Returns debugging information in the form of a dictionary."""
        return self._logs
