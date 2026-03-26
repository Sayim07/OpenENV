from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


class SenderTier(str, Enum):
    """
    Tier of the email sender.
    """
    EXECUTIVE = "executive"
    MANAGER = "manager"
    PEER = "peer"
    EXTERNAL = "external"
    SPAM = "spam"


class ActionType(str, Enum):
    """
    Type of action to take on an email.
    """
    ARCHIVE = "archive"
    LABEL_URGENT = "label_urgent"
    LABEL_DELEGATE = "label_delegate"
    DRAFT_REPLY = "draft_reply"
    ESCALATE = "escalate"
    FLAG_SPAM = "flag_spam"
    SNOOZE = "snooze"
    NO_OP = "no_op"


class Observation(BaseModel):
    """
    Observation model representing an incoming email.

    Attributes:
        message_id (str): Unique identifier for the message.
        subject (str): Subject of the email. Maximum 120 characters.
        sender_email (EmailStr): Email address of the sender.
        sender_tier (SenderTier): Categorized tier of the sender.
        body_snippet (str): Snippet of the email body. Maximum 500 characters.
        thread_depth (int): Depth of the email thread. Must be >= 0.
        has_attachment (bool): Whether the email has an attachment.
        received_at (datetime): Timestamp when the email was received.
        urgency_signal (float): Signal indicating urgency, from 0.0 to 1.0.
        inbox_remaining (int): Number of emails remaining in the inbox. Must be >= 0.
        step_budget_remaining (int): Remaining steps allowed for the agent. Must be >= 0.
        context (dict[str, Any]): Additional context for the observation. Defaults to {}.
    """
    message_id: str
    subject: str = Field(..., max_length=120)
    sender_email: EmailStr
    sender_tier: SenderTier
    body_snippet: str = Field(..., max_length=500)
    thread_depth: int = Field(..., ge=0)
    has_attachment: bool
    received_at: datetime
    urgency_signal: float = Field(..., ge=0.0, le=1.0)
    inbox_remaining: int = Field(..., ge=0)
    step_budget_remaining: int = Field(..., ge=0)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v: str) -> str:
        """
        Validates that the subject does not exceed 120 characters.

        Args:
            v (str): The subject string.

        Returns:
            str: The validated subject string.

        Raises:
            ValueError: If the subject is longer than 120 characters.
        """
        if len(v) > 120:
            raise ValueError("subject must be exactly or fewer than 120 characters")
        return v

    @field_validator("body_snippet")
    @classmethod
    def validate_body_snippet(cls, v: str) -> str:
        """
        Validates that the body_snippet does not exceed 500 characters.

        Args:
            v (str): The body snippet string.

        Returns:
            str: The validated body snippet string.

        Raises:
            ValueError: If the snippet is longer than 500 characters.
        """
        if len(v) > 500:
            raise ValueError("body_snippet must be exactly or fewer than 500 characters")
        return v

    @field_validator("thread_depth", "inbox_remaining", "step_budget_remaining")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """
        Validates that the given field is non-negative.

        Args:
            v (int): The integer value to validate.

        Returns:
            int: The validated integer.

        Raises:
            ValueError: If the integer is less than 0.
        """
        if v < 0:
            raise ValueError("value must be greater than or equal to 0")
        return v

    @field_validator("urgency_signal")
    @classmethod
    def validate_urgency_signal(cls, v: float) -> float:
        """
        Validates that the urgency signal is between 0.0 and 1.0 inclusive.

        Args:
            v (float): The urgency signal.

        Returns:
            float: The validated urgency signal.

        Raises:
            ValueError: If the signal is not between 0.0 and 1.0.
        """
        if not (0.0 <= v <= 1.0):
            raise ValueError("urgency_signal must be between 0.0 and 1.0")
        return v


class InvalidActionError(Exception):
    """
    Exception raised when an action or action payload is invalid.
    """
    pass


class Action(BaseModel):
    """
    Action model representing an action to take on an email.

    Attributes:
        action_type (ActionType): The type of the action.
        payload (dict[str, Any]): Additional information required for the action.
    """
    action_type: ActionType
    payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_action_payload(self) -> "Action":
        """
        Validates the payload based on the specific action_type.

        Returns:
            Action: The validated action instance.

        Raises:
            ValueError: If action_type is LABEL_DELEGATE and payload misses 'delegate_to',
                or if action_type is DRAFT_REPLY and payload misses 'reply_body'.
        """
        if self.action_type == ActionType.LABEL_DELEGATE:
            if "delegate_to" not in self.payload:
                raise ValueError("Payload missing 'delegate_to' for action_type LABEL_DELEGATE.")
        elif self.action_type == ActionType.DRAFT_REPLY:
            if "reply_body" not in self.payload:
                raise ValueError("Payload missing 'reply_body' for action_type DRAFT_REPLY.")
        return self


class State(BaseModel):
    """
    State model capturing the full internal episode state for checkpointing.

    Attributes:
        observations (list[Observation]): Sequence of past observations.
        actions (list[Action]): Sequence of past actions taken.
        current_step (int): The current step index in the episode.
        done (bool): Whether the episode has reached a terminal state.
    """
    observations: list[Observation] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    current_step: int = Field(default=0, ge=0)
    done: bool = Field(default=False)
