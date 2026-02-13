import pytest


@pytest.fixture
def sample_text() -> str:
    return (
        "This Non-Disclosure Agreement (NDA) is entered into by and between "
        "Party A and Party B. The term of this agreement shall be 2 years from "
        "the effective date. Either party may terminate this agreement with "
        "30 days written notice."
    )


@pytest.fixture
def sample_chunks() -> list[str]:
    return [
        "This Non-Disclosure Agreement is entered into by Party A and Party B.",
        "The term of this agreement shall be 2 years from the effective date.",
        "Either party may terminate this agreement with 30 days written notice.",
        "Confidential information includes all technical and business information.",
        "The receiving party shall not disclose confidential information to third parties.",
    ]
