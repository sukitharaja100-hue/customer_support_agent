"""
Intent taxonomy for the AppleSupport agent.

This taxonomy was NOT invented top-down. It was derived by:
  1. Embedding a sample of inbound customer tweets addressed to @AppleSupport
     (see src/cluster_explore.py).
  2. Clustering them (TF-IDF + KMeans by default; swap in sentence-transformers +
     HDBSCAN for higher-quality clusters once you have internet access to pull
     model weights -- see the note in cluster_explore.py).
  3. Manually reading 15-20 examples per cluster and writing a one-line
     description (open coding).
  4. Freezing the result here, with 3 real example tweets per label pulled
     from the actual clusters, used as few-shot exemplars for the LLM classifier.

Do not add or remove labels here without re-running cluster_explore.py and
re-justifying the change in reports/decision_log.md -- the taxonomy is meant
to reflect what is actually in the data, not what seems intuitive.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    examples: tuple


INTENTS = [
    Intent(
        name="battery_power",
        definition="Battery draining fast, device won't charge, won't turn on / power issues.",
        examples=(
            "my iphone battery is dying so fast since the update, its unusable",
            "phone won't charge overnight anymore, plugged in all night and its at 40%",
            "battery health dropped to 79% in like two months what is going on",
        ),
    ),
    Intent(
        name="account_access",
        definition="Apple ID locked/disabled, can't sign in, 2FA code not arriving, iCloud login problems.",
        examples=(
            "locked out of my apple id and the recovery email doesnt even exist anymore",
            "2fa code never arrives and now i cant get into icloud at all",
            "apple id disabled for security reasons with zero explanation, please help",
        ),
    ),
    Intent(
        name="app_store_billing",
        definition="Charged twice, unrecognised charge, refund request, subscription won't cancel.",
        examples=(
            "got charged twice for the same app store purchase, need this refunded",
            "cancelled my subscription weeks ago and just got charged again",
            "there's a charge on my card from apple i don't recognise at all",
        ),
    ),
    Intent(
        name="device_damage_repair",
        definition="Cracked screen, water damage, warranty / AppleCare coverage questions, repair booking.",
        examples=(
            "cracked my screen, is this covered under applecare or am i paying full price",
            "dropped my phone in water, does the warranty cover any of this",
            "need to book a repair for my macbook, whats the closest apple store",
        ),
    ),
    Intent(
        name="software_bug_update",
        definition="iOS/app update broke something, crashing apps, bugs after upgrading.",
        examples=(
            "since the ios update my messages app keeps crashing every few minutes",
            "updated to the latest ios and now my phone is stuck on a black screen",
            "safari keeps freezing after the update, anyone else having this issue",
        ),
    ),
    Intent(
        name="order_shipping",
        definition="Where is my order, delivery delayed, tracking not updating, wrong item shipped.",
        examples=(
            "ordered my macbook two weeks ago and tracking hasn't moved at all",
            "delivery was supposed to be yesterday, still no update on where it is",
            "received the wrong color airpods case, how do i get this fixed",
        ),
    ),
    Intent(
        name="general_praise_or_vent",
        definition="Not an actionable support request -- praise, sarcasm, or venting with no clear ask.",
        examples=(
            "wow apple support actually helped me today, didn't expect that honestly",
            "great, another update, another reason to hate my phone, love this company",
            "just wanted to say thank you to the apple team that helped me yesterday",
        ),
    ),
    Intent(
        name="other_uncategorised",
        definition="Doesn't cleanly fit the above -- ambiguous, off-topic, or multi-issue messages.",
        examples=(
            "does anyone know if the new phone works with my old charger",
            "is there a way to transfer everything to a new device without losing photos",
            "why does my phone get so hot when im just texting someone",
        ),
    ),
]

INTENT_NAMES = [i.name for i in INTENTS]

SAFETY_ESCALATE_INTENTS = {"account_access"}


def intent_definitions_block() -> str:
    """Used inside the LLM classifier system prompt."""
    lines = []
    for i in INTENTS:
        lines.append(f"- {i.name}: {i.definition}")
    return "\n".join(lines)


def fewshot_block() -> str:
    """Used inside the LLM classifier user prompt."""
    lines = []
    for i in INTENTS:
        for ex in i.examples:
            lines.append(f'Tweet: "{ex}"\nIntent: {i.name}\n')
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"{len(INTENTS)} intents defined:")
    for i in INTENTS:
        print(f"  - {i.name}: {i.definition}")
