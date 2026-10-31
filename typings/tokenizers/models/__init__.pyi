# tokenizers 0.22.2 models.pyi signatures; its generated package reexports lack types.
class Model: ...

class WordLevel(Model):
    def __new__(
        cls, vocab: dict[str, int] | None = None, unk_token: str = "<unk>"
    ) -> WordLevel:
        """Construct a word-level tokenizer model."""
        ...
